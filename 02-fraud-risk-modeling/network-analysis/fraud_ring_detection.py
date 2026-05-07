"""
Fraud Ring Detection via Graph Analysis
=========================================

Surfaces fraud rings by building a bipartite graph between accounts and
shared identity attributes (devices, IPs, phones, emails, addresses), then
projecting onto an account-account graph and running community detection.

The signal: legitimate customers rarely share device + IP + phone +
address with multiple other "different" customers. Fraud rings, by
contrast, recycle attributes constantly — one device files five
applications under five different names.

This is a stylized synthetic example: we generate accounts, sprinkle in
some fraud rings with shared attributes, build the graph, and show how
connected components + community-level fraud concentration surface them.

Run:
    python fraud_ring_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-shared-utilities"))

from plotting import PALETTE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CHARTS_DIR = Path(__file__).resolve().parent / "charts"

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Synthetic accounts + planted fraud rings
# ---------------------------------------------------------------------------
def generate_accounts(n_legit: int = 5_000, n_rings: int = 25, ring_size: int = 8):
    """Generate accounts. Each ring shares device/IP/phone across members."""
    rows = []

    # Legitimate accounts: each has its own unique device, IP, phone, email
    for i in range(n_legit):
        rows.append({
            "account_id": f"L{i:05d}",
            "device_id": f"D_legit_{i}",
            "ip_address": f"IP_legit_{i % 4500}",   # a little IP sharing is normal
            "phone": f"P_legit_{i}",
            "email_domain": RNG.choice(["gmail", "yahoo", "outlook", "icloud"]),
            "is_fraud": 0,
        })

    # Fraud rings: ~ring_size accounts each, sharing 1-3 attributes
    for r in range(n_rings):
        shared_device = f"D_ring{r}_a"
        shared_ip = f"IP_ring{r}_a"
        shared_phone = f"P_ring{r}_a"
        for k in range(ring_size):
            # The ring has SOME variation per member but shares core attributes
            rows.append({
                "account_id": f"F{r:03d}_{k:02d}",
                "device_id": shared_device if k % 2 == 0 else f"D_ring{r}_b",
                "ip_address": shared_ip,                       # all share IP
                "phone": shared_phone if k < ring_size // 2 else f"P_ring{r}_{k}",
                "email_domain": "tempmail",                     # tells, but legit users use it too
                "is_fraud": 1,
            })

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_account_graph(df: pd.DataFrame, attributes: list[str]) -> nx.Graph:
    """
    Build an account-account graph. An edge exists if two accounts share
    at least one value of any of the given attributes. Edge weight =
    number of attributes shared.
    """
    g = nx.Graph()
    for acc, fraud in zip(df.account_id, df.is_fraud):
        g.add_node(acc, is_fraud=int(fraud))

    # Group accounts by attribute value, link any pair within the same group
    edges: dict[tuple[str, str], int] = {}
    for attr in attributes:
        for value, group in df.groupby(attr):
            accs = list(group.account_id)
            if len(accs) < 2:
                continue
            # Skip extremely common values — they create huge dense
            # subgraphs that aren't informative (e.g., "gmail" email domain)
            if len(accs) > 100:
                continue
            for i in range(len(accs)):
                for j in range(i + 1, len(accs)):
                    key = tuple(sorted([accs[i], accs[j]]))
                    edges[key] = edges.get(key, 0) + 1

    for (a, b), w in edges.items():
        g.add_edge(a, b, weight=w)
    return g


# ---------------------------------------------------------------------------
# Score communities
# ---------------------------------------------------------------------------
def score_components(g: nx.Graph) -> pd.DataFrame:
    """
    For each connected component, compute size and observed fraud rate.
    A component that is large AND has a high fraud rate is a fraud ring
    candidate.
    """
    rows = []
    for cid, component in enumerate(nx.connected_components(g)):
        nodes = list(component)
        if len(nodes) < 2:
            continue
        fraud_flags = [g.nodes[n]["is_fraud"] for n in nodes]
        rows.append({
            "component_id": cid,
            "size": len(nodes),
            "n_fraud": int(sum(fraud_flags)),
            "fraud_rate": np.mean(fraud_flags),
            "members_sample": ", ".join(nodes[:5]) + ("…" if len(nodes) > 5 else ""),
        })
    return pd.DataFrame(rows).sort_values(["fraud_rate", "size"], ascending=False)


# ---------------------------------------------------------------------------
# Account-level risk score
# ---------------------------------------------------------------------------
def account_risk_score(g: nx.Graph) -> pd.DataFrame:
    """
    Per-account features:
        - degree                 : number of distinct accounts sharing an attribute
        - weighted degree        : sum of edge weights (more attributes shared = riskier)
        - component_size         : size of connected component
        - component_fraud_rate   : observed fraud rate in the component
                                   (in production, use historical fraud labels of OTHER accounts)
    """
    components = list(nx.connected_components(g))
    node_to_comp = {}
    for cid, comp in enumerate(components):
        for n in comp:
            node_to_comp[n] = cid
    comp_size = {cid: len(c) for cid, c in enumerate(components)}
    comp_fraud_rate = {
        cid: np.mean([g.nodes[n]["is_fraud"] for n in c])
        for cid, c in enumerate(components)
    }

    rows = []
    for n in g.nodes:
        cid = node_to_comp[n]
        rows.append({
            "account_id": n,
            "is_fraud": g.nodes[n]["is_fraud"],
            "degree": g.degree(n),
            "weighted_degree": g.degree(n, weight="weight"),
            "component_id": cid,
            "component_size": comp_size[cid],
            "component_fraud_rate": comp_fraud_rate[cid],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating synthetic accounts with planted fraud rings…")
    df = generate_accounts(n_legit=5_000, n_rings=25, ring_size=8)
    print(f"  Total accounts : {len(df):,}")
    print(f"  Fraud accounts : {df.is_fraud.sum():,} ({df.is_fraud.mean():.2%})")

    print("\nBuilding account-account graph on shared attributes…")
    attrs = ["device_id", "ip_address", "phone"]
    g = build_account_graph(df, attrs)
    print(f"  Nodes : {g.number_of_nodes():,}")
    print(f"  Edges : {g.number_of_edges():,}")

    # Component-level analysis
    print("\nTop 15 connected components by fraud rate:")
    comp_df = score_components(g)
    print(comp_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nComponents with >= 80% fraud rate and size >= 4 :")
    suspicious = comp_df[(comp_df.fraud_rate >= 0.80) & (comp_df["size"] >= 4)]
    print(f"  Count          : {len(suspicious)}")
    print(f"  Fraud captured : {suspicious.n_fraud.sum()} of {df.is_fraud.sum()} "
          f"({suspicious.n_fraud.sum() / max(df.is_fraud.sum(), 1):.2%})")
    print(f"  Total accounts in suspicious components : {suspicious['size'].sum()}")

    # Account-level features ready to merge into the application/transaction model
    print("\nGenerating account-level graph features…")
    feat = account_risk_score(g)
    print(f"  Generated {len(feat):,} rows")
    print("\n  Mean feature values, fraud vs. legit:")
    print(feat.groupby("is_fraud")[
        ["degree", "weighted_degree", "component_size", "component_fraud_rate"]
    ].mean().round(3).to_string())

    # The graph features should massively separate the two populations:
    print("\nFraud accounts have order-of-magnitude higher degree and component size.")
    print("In production these would feed the application-fraud model as additional features.")

    # ---- charts ----
    CHARTS_DIR.mkdir(exist_ok=True)
    print(f"\nSaving charts to {CHARTS_DIR}/ …")

    # Component fraud-rate distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    big_comps = comp_df[comp_df["size"] >= 2]
    ax.scatter(big_comps["size"], big_comps.fraud_rate,
               s=big_comps["size"] * 8, alpha=0.6,
               c=[PALETTE["bad"] if r >= 0.8 else PALETTE["accent"]
                  for r in big_comps.fraud_rate],
               edgecolor="white")
    ax.axhline(0.80, color=PALETTE["bad"], ls="--", lw=1, label="High-risk threshold (80%)")
    ax.set_xlabel("Component size (# accounts)")
    ax.set_ylabel("Fraud rate within component")
    ax.set_title("Network Analysis — Connected Component Fraud Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "component_fraud_rate.png")
    plt.close(fig)

    # Feature separation: degree by fraud status
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    legit_deg = feat[feat.is_fraud == 0].degree
    fraud_deg = feat[feat.is_fraud == 1].degree
    axes[0].hist([legit_deg, fraud_deg], bins=15,
                 label=["Legit", "Fraud"], color=[PALETTE["primary"], PALETTE["bad"]],
                 edgecolor="white")
    axes[0].set_xlabel("Node degree (# linked accounts)")
    axes[0].set_ylabel("Account count")
    axes[0].set_title("Degree by Fraud Status")
    axes[0].legend()

    legit_cs = feat[feat.is_fraud == 0].component_size
    fraud_cs = feat[feat.is_fraud == 1].component_size
    axes[1].hist([legit_cs, fraud_cs], bins=15,
                 label=["Legit", "Fraud"], color=[PALETTE["primary"], PALETTE["bad"]],
                 edgecolor="white")
    axes[1].set_xlabel("Component size")
    axes[1].set_ylabel("Account count")
    axes[1].set_title("Component Size by Fraud Status")
    axes[1].legend()

    fig.suptitle("Network Analysis — Graph Features Separate Fraud from Legit",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "graph_features_separation.png", bbox_inches="tight")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

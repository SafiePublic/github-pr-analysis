#
# (c) 2024 Safie Inc.
#
# Notice:
#   No part of this file may be reproduced, stored
#   in a retrieval system, or transmitted, in any form, or by any means,
#   electronic, mechanical, photocopying, recording, or otherwise,
#   without the prior consent of Safie Inc.
#
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

import config as cfg
import matplotlib.pyplot as plt  # 250 ms
import matplotlib.ticker as ticker
import networkx as nx  # 70 ms
import numpy as np  # 40 ms
import plotly.graph_objects as go
import requests
import seaborn as sns  # 780 ms
from matplotlib.colors import rgb2hex
from tqdm import tqdm


def validate_date(date_string: str) -> None:
    pattern = r"^\d{4}-\d{2}-\d{2}$"
    if not re.match(pattern, date_string):
        print("date format must be yyyy-mm-dd")
        sys.exit(1)


def validate_period(from_date: str, to_date: str) -> None:
    if from_date > to_date:
        print("from_date must be earlier than to_date")
        sys.exit(1)


def search_pr_by_authors(usernames: list[str], from_date: str, to_date: str, token: str) -> dict:
    headers = {
        "Accept": "application/vnd.github.text-match+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    usernames = usernames.copy()
    usernames.insert(0, "")  # Insert empty string to add prefix for the first element

    query_webapp = "type:pr"
    query_webapp += " author:".join(usernames)
    query_webapp += f" created:{from_date}..{to_date}"
    print("Search query for webapp: ")
    print(query_webapp)

    query_rest = "type:pr"
    query_rest += "+author:".join(usernames)
    query_rest += f"+created:{from_date}..{to_date}&sort=created&order=desc&per_page=100"
    url = f"https://api.github.com/search/issues?q={query_rest}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(response)
        sys.exit(1)

    pulls = response.json()

    if False:  # for debug
        search_cache = f"search_result_{from_date}_{to_date}.json"
        json.dump(pulls, open(search_cache, "w"), indent=2)

    if len(pulls["items"]) < pulls["total_count"] or pulls["incomplete_results"]:
        print("Cannot retrive all pull requests (should be <= 100)")
        sys.exit(1)

    return pulls


def check_pr_update(item: dict, search_api_cache: dict) -> bool:
    url = item["html_url"]
    if url in search_api_cache:
        updated_at = search_api_cache[url]
        if item["updated_at"] == updated_at:
            return False
    return True


def get_requested_reviewers(
    owner: str, repository: str, pr_number: int, token: str, pulls_api_cache: dict, reflesh: bool
) -> list[str]:
    # Use GET /repos/{owner}/{repo}/pulls/{pull_number}/requested_reviewers

    url = f"https://api.github.com/repos/{owner}/{repository}/pulls/{pr_number}/requested_reviewers"

    if url not in pulls_api_cache or reflesh:
        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(response)
            sys.exit(1)
        response_json = response.json()
        pulls_api_cache[url] = response_json
    else:
        response_json = pulls_api_cache[url]

    reviewers = []
    for reviewer in response_json["users"]:
        reviewers.append(reviewer["login"])
    return reviewers


def get_completed_reviewers(
    owner: str,
    repository: str,
    pr_number: int,
    author: str,
    requested: list,
    token: str,
    pulls_api_cache: dict,
    reflesh: bool,
) -> list[str]:
    # Use GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews

    url = f"https://api.github.com/repos/{owner}/{repository}/pulls/{pr_number}/reviews"

    if url not in pulls_api_cache or reflesh:
        headers = {
            "Accept": "application/vnd.github.text-match+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(response)
            sys.exit(1)
        response_json = response.json()
        pulls_api_cache[url] = response_json
    else:
        response_json = pulls_api_cache[url]

    reviewers = []
    for review in response_json:
        reviewers.append(review["user"]["login"])
    reviewers = list(set(reviewers))  # Remove duplicates

    if author in reviewers:  # Remove self comment
        reviewers.remove(author)

    for reviewer in requested:
        if reviewer in reviewers:
            reviewers.remove(reviewer)  # Remove re-requested reviewer from reviewed reviewers

    return reviewers


def update_data(
    data: np.ndarray, repo_name: str, pr_number: int, author: str, authors: list, requested: list, completed: list
) -> None:
    author_index = authors.index(author)
    for reviewer in requested:
        try:
            reviewer_index = authors.index(reviewer)
        except ValueError:
            print(f"Review requested to other group member: {reviewer} in {repo_name} #{pr_number}")
            continue
        data[0][author_index][reviewer_index] += 1

    for reviewer in completed:
        try:
            reviewer_index = authors.index(reviewer)
        except ValueError:
            print(f"Reviewed by other group member: {reviewer} in {repo_name} #{pr_number}")
            continue
        data[1][author_index][reviewer_index] += 1


def plot_color_map(data: np.ndarray, authors: list[str], from_date: str, to_date: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap="flare", xticklabels=authors, yticklabels=authors, vmin=0, vmax=10)
    plt.title(f"Author-reviewer heatmap (from {from_date} to {to_date})")
    plt.xlabel("reviewer")
    plt.ylabel("author")
    plt.xticks(rotation=75)
    plt.tight_layout()
    filename = f"{from_date}_{to_date}_heatmap.png"
    plt.savefig(filename)
    print(f"Heatmap was saved as {filename}")


def plot_stacked_column_chart(
    data: np.ndarray, author_count: np.ndarray, authors: list[str], from_date: str, to_date: str
) -> None:
    requested_count = np.sum(data[0], axis=0)
    completed_count = np.sum(data[1], axis=0)
    print(
        "authors count: ",
        author_count,
        ", review-requested count: ",
        requested_count,
        ", review-completed count: ",
        completed_count,
    )

    palette = sns.color_palette("muted")
    plt.figure(figsize=(7.5, 6))
    plt.bar(authors, completed_count, label="review-completed", color=palette[0])
    plt.bar(authors, requested_count, bottom=completed_count, label="review-requested", color=palette[3])
    plt.bar(authors, author_count, bottom=completed_count + requested_count, label="author", color=palette[8])
    plt.title(f"Author-reviewer count (from {from_date} to {to_date})")
    plt.xlabel("member")
    plt.ylabel("count")
    plt.xticks(rotation=75)
    plt.grid(which="major", linestyle=":", axis="y")  # Major grid lines
    plt.legend()
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Set integer y-axis
    plt.tight_layout()
    filename = f"{from_date}_{to_date}_stacked_column_chart.png"
    plt.savefig(filename)
    print(f"Stacked column chart was saved as {filename}")


def adjust_layout(layout: dict, is_narrow: bool = False) -> dict:
    min_distance = 0.3
    if is_narrow:
        min_distance = 0.2

    adjusted_layout = {}
    for node, pos in layout.items():
        new_pos = pos.copy()
        for other_node, other_pos in layout.items():
            if node == other_node:
                continue
            distance = np.linalg.norm(new_pos - other_pos)
            if distance >= min_distance:
                continue
            direction = new_pos - other_pos
            if abs(direction[0]) < 0.01 or abs(direction[1]) < 0.01:  # Fix eigenvalue degeneration
                direction = np.random.rand(2) - 0.5
            direction /= np.linalg.norm(direction)
            new_pos += direction * (min_distance - distance) * 0.1
        adjusted_layout[node] = new_pos
    return adjusted_layout


def plot_network_graph(data: np.ndarray, authors: list[str], from_date: str, to_date: str, narrow: bool) -> None:
    data = data + data.T  # Create symmetric matrix
    g = nx.from_numpy_array(data, create_using=nx.Graph)
    labels = {i: author for i, author in enumerate(authors)}
    edge_widths = [d["weight"] * 2 for (u, v, d) in g.edges(data=True)]
    node_sizes = [sum([g[u][v]["weight"] for v in g.neighbors(u)]) * 100 for u in g.nodes()]
    layout = nx.spectral_layout(g)
    for _ in range(100):
        layout = adjust_layout(layout, narrow)  # Adjust node positions to avoid overlap

    # Plot
    plt.figure(figsize=(10, 8))
    plt.title(f"Author-reviewer network graph (from {from_date} to {to_date})")
    nx.draw(
        g,
        pos=layout,
        with_labels=True,
        labels=labels,
        node_color="lightblue",
        edge_color="lightgreen",
        node_size=node_sizes,
        width=edge_widths,
        font_size=12,
    )
    plt.margins(0.1)
    plt.tight_layout()
    filename = f"{from_date}_{to_date}_network_graph.png"
    plt.savefig(filename)
    print(f"Network graph was saved as {filename}")


def plot_sankey_diagram(data: np.ndarray, authors: list[str], from_date: str, to_date: str) -> None:
    # Calculate link flows
    sources = []
    targets = []
    flows = []
    num_authors = len(authors)
    for i in range(num_authors):
        for j in range(num_authors):
            sources.append(i)
            targets.append(num_authors + j)
            flows.append(data[i][j])
    link = dict(source=sources, target=targets, value=flows)

    # Calculate node x-positions
    authors_count = np.sum(data, axis=1)
    reviewers_count = np.sum(data, axis=0)
    num_nonzero_authors = np.count_nonzero(authors_count)
    num_nonzero_reviewers = np.count_nonzero(reviewers_count)
    x_positions = [0.001 for _ in range(num_nonzero_authors)] + [0.999 for _ in range(num_nonzero_reviewers)]

    # Calculate node y-positions
    authors_positions = np.linspace(0.1, 0.9, num_nonzero_authors)
    reviewers_positions = np.linspace(0.1, 0.9, num_nonzero_reviewers)
    y_positions = np.concatenate([authors_positions, reviewers_positions])

    # Create color map
    palette = sns.color_palette("muted", 11)
    colors = [rgb2hex(palette[i]) for i in range(num_authors)] * 2
    node = dict(label=authors + authors, x=x_positions, y=y_positions, color=colors)

    # Plot
    sankey = go.Sankey(link=link, node=node, arrangement="snap")
    fig = go.Figure(sankey)
    fig.update_layout(title_text=f"Author-reviewer sankey diagram (from {from_date} to {to_date})", width=800, height=640)
    filename = f"{from_date}_{to_date}_sankey_diagram.png"
    fig.write_image(filename, scale=1.5)
    print(f"Sankey diagram was saved as {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="analyze GitHub pull requests")
    parser.add_argument("--from_date", type=str, metavar="DATE", help="search period start in yyyy-mm-dd")
    parser.add_argument("--to_date", type=str, metavar="DATE", help="search period end in yyyy-mm-dd")
    parser.add_argument("-w", "--week", action="store_true", help="aggragate latest 1 week")
    parser.add_argument("-n", "--narrow", action="store_true", help="narrow node distance in network graph")
    args = parser.parse_args()

    if args.to_date is None:
        args.to_date = datetime.now().strftime("%Y-%m-%d")  # Set today
    else:
        validate_date(args.to_date)

    # Check mutually exclusive options
    if args.from_date is not None and args.week:
        print("--from_date and --week are mutually exclusive")
        sys.exit(1)

    if args.from_date is not None:
        validate_date(args.from_date)
    elif args.week:  # Set 1 week before
        args.from_date = (datetime.strptime(args.to_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    else:  # Set 30 days before
        args.from_date = (datetime.strptime(args.to_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    validate_period(args.from_date, args.to_date)
    print(f"Aggregation period: from {args.from_date} to {args.to_date}")

    return args


def main():
    args = parse_args()
    token = cfg.github_token
    authors = cfg.authors

    # Load search API cache
    search_api_cache_filename = "search_api_cache.json"
    if os.path.exists(search_api_cache_filename):
        with open(search_api_cache_filename, "r") as f:
            search_api_cache = json.load(f)
    else:
        search_api_cache = {}

    # Search pull requests
    pulls = search_pr_by_authors(authors, args.from_date, args.to_date, token)  # Rate limit: 10 times per minute
    num_pr_tot = pulls["total_count"]
    print(f"# searched pull requests: {num_pr_tot}")

    # Load pulls API cache
    pulls_api_cache_filename = "pulls_api_cache.json"
    if os.path.exists(pulls_api_cache_filename):
        with open(pulls_api_cache_filename, "r") as f:
            pulls_api_cache = json.load(f)
    else:
        pulls_api_cache = {}

    # Calculate author-reviewer matrix
    print(f"Call GitHub REST API {2 * num_pr_tot} times. Check GitHub rate limit for more details. Use cache if available.")
    n = len(authors)
    data = np.zeros((2, n, n), dtype=int)  # 1st-axis: requested/reviewed, 2nd-axis: author, 3rd-axis: reviewer
    author_count = np.zeros(n, dtype=int)
    items = pulls["items"]
    num_items = len(items)
    for i in tqdm(range(num_items)):
        item = items[i]
        owner = item["repository_url"].split("/")[-2]
        repo_name = item["repository_url"].split("/")[-1]
        pr_number = item["number"]
        author = item["user"]["login"]
        reflesh = check_pr_update(item, search_api_cache)
        requested = get_requested_reviewers(owner, repo_name, pr_number, token, pulls_api_cache, reflesh)
        completed = get_completed_reviewers(owner, repo_name, pr_number, author, requested, token, pulls_api_cache, reflesh)
        search_api_cache[item["html_url"]] = item["updated_at"]  # Update timestamp
        author_count[authors.index(author)] += 1
        update_data(data, repo_name, pr_number, author, authors, requested, completed)
    json.dump(pulls_api_cache, open(pulls_api_cache_filename, "w"), indent=2)
    json.dump(search_api_cache, open(search_api_cache_filename, "w"), indent=2)
    print("Author-reviewer matrix (review-requested, review-completed): ")
    print(data)

    # Visualize
    plot_color_map(data[0] + data[1], authors, args.from_date, args.to_date)
    plot_stacked_column_chart(data, author_count, authors, args.from_date, args.to_date)
    plot_network_graph(data[0] + data[1], authors, args.from_date, args.to_date, args.narrow)
    plot_sankey_diagram(data[0] + data[1], authors, args.from_date, args.to_date)


if __name__ == "__main__":
    main()

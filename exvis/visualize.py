import argparse
import gzip
import math
import tarfile

import matplotlib.pyplot as plt
import networkx as nx


class Constraint:
    """
    A constraint is a relation between variables.
    """

    def __init__(self):
        self.variables = []


class Variable:
    """
    A variable is a quantity that can be changed.
    """

    def __init__(self, id):
        self.id = id
        self.constraints = []


class Model:
    """
    A model is a collection of variables and constraints.
    """

    def __init__(self):
        self.variables = {}
        self.constraints = []

    def create_variable(self, id: str) -> Variable:
        if id in self.variables:
            return self.variables[id]
        self.variables[id] = Variable(id)
        return self.variables[id]

    def create_constraint_relation(self, variables: list[str]) -> None:
        vars = [self.create_variable(id) for id in variables]
        constraint = Constraint()
        constraint.variables = vars
        self.constraints.append(constraint)
        for var in vars:
            var.constraints.append(constraint)


def is_variable(word: str) -> bool:
    """
    Check if a word is a variable.

    A name should be no longer than 255 characters, and to avoid confusing the LP parser,
    it can not begin with a number or any of the characters +, -, *, ^, <, >, =, (, ),
    [, ], ,, or :. For similar reasons, a name should not contain any of the characters +,
    -, *, ^, or :.
    """
    if len(word) > 255 or len(word) == 0:
        return False
    if word[0].isdigit():
        return False
    if word[0] in ["+", "-", "*", "^", "<", ">", "=", "(", ")", "[", "]", ",", ":"]:
        return False
    if any(c in ["+", "-", "*", "^", ":"] for c in word):
        return False
    return True


def read_lp(file: str) -> Model:
    """
    Read a linear program from a file in LP format.
    """
    # Objective section start
    section_obj = {"minimize", "maximize", "minimum", "maximum", "min", "max"}
    # Constraint section start
    section_con = {"subject to", "such that", "st", "s.t."}

    # Open file (compressed or not)
    if file.endswith(".gz"):
        f = gzip.open(file, "rt")
    elif file.endswith(".tar.gz"):
        tar = tarfile.open(file, "r:gz")
        f = tar.extractfile(tar.getmembers()[0])
    else:
        f = open(file)

    model = Model()
    current_section = None
    # Read line by line
    for line in f:
        # Ignore comments
        if line.startswith("\\"):
            continue
        stripped_lower = line.strip().lower()
        # Ignore empty lines
        if len(stripped_lower) == 0:
            continue
        # The line is a section header if it starts with some text (not whitespace or a comment)
        is_header = line[0].isalpha()
        # If the line does not start with whitespace, it is a section header
        if is_header:
            if stripped_lower in section_obj:
                current_section = "objective"
                continue
            if stripped_lower in section_con:
                current_section = "constraint"
                continue
            current_section = None
            continue
        # Only read variable relations from constraint and objective sections
        if current_section != "constraint" and current_section != "objective":
            continue
        # Proceed with the cleaned up line
        line = stripped_lower
        # Remove trailing comments
        if "\\" in line:
            line = line[: line.index("\\")]
        # Remove name/label (if any)
        if ":" in line:
            line = line[line.index(":") + 1 :]
        # Parse variable relations
        variables = [w for w in line.split() if is_variable(w)]
        if len(variables) > 0:
            model.create_constraint_relation(variables)

    # Close file
    f.close()

    return model


def read_mps(file: str) -> Model:
    """
    Read a linear program from a file in MPS format.
    """

    # Open file (compressed or not)
    if file.endswith(".gz"):
        f = gzip.open(file, "rt")
    elif file.endswith(".tar.gz"):
        tar = tarfile.open(file, "r:gz")
        f = tar.extractfile(tar.getmembers()[0])
    else:
        f = open(file)

    rows = {}
    is_columns = False
    # Read line by line
    for line in f:
        # We are only interested in the COLUMNS section
        if line.startswith("COLUMNS"):
            is_columns = True
            continue
        if not is_columns:
            continue
        # Ignore empty lines
        if len(line.strip()) == 0:
            continue
        # If line does not start with whitespace, it is the next section
        if not line[0].isspace():
            break
        # Parse NNZ info
        content = [w for w in line.split() if w != ""]
        # Ignore markers
        if "MARKER" in content[1]:
            continue
        row = content[0]
        for i in range(1, len(content), 2):  # 2 because we have a variable and a value
            col = content[i]
            if row not in rows:
                rows[row] = []
            rows[row].append(col)

    # Close file
    f.close()

    model = Model()
    for row in rows.values():
        model.create_constraint_relation(row)
    return model


def plot(
    model: Model,
    file: str,
    log: bool = False,
    dark: bool = False,
    weighted: bool = False,
) -> None:
    """
    Plot a model as a graph.
    """
    # Create a graph
    G = nx.Graph()
    # Add nodes
    for var in model.variables.values():
        G.add_node(var.id)
    # Determine edges (variables that appear in the same constraint, avoid duplicates)
    edges = {}
    for constraint in model.constraints:
        for i in range(len(constraint.variables)):
            for j in range(i + 1, len(constraint.variables)):
                if constraint.variables[i].id < constraint.variables[j].id:
                    edge = (constraint.variables[i].id, constraint.variables[j].id)
                else:
                    edge = (constraint.variables[j].id, constraint.variables[i].id)
                if edge not in edges:
                    edges[edge] = 1
                else:
                    edges[edge] += 1
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edges[edge] if weighted else 1)
    # Layout graph
    print("Computing layout...")
    pos = nx.spring_layout(G)
    # Color nodes by betweenness centrality
    colors = nx.betweenness_centrality(G)
    colors = [colors[node] for node in G.nodes()]
    # Apply log scale
    if log:
        colors = [math.log(c + 1) for c in colors]
    # Determine node size by number of nodes
    size_factor = 1 - min(1, len(G.nodes()) / 4000)
    n_size = 25 + size_factor * 300
    e_width = 0.1 + size_factor * 1
    # Plot graph
    fig, ax = plt.subplots(figsize=(30, 30))
    nx.draw_networkx_nodes(G, pos, node_size=n_size, node_color=colors, cmap="plasma")
    nx.draw_networkx_edges(G, pos, width=e_width, alpha=0.3, edge_color="#ffffff" if dark else "#000000")
    if dark:
        ax.set_facecolor("black")
        ax.axis("off")
        fig.set_facecolor("black")
    # Save figure
    print(f"Saving graph to {file}...")
    plt.savefig(file, bbox_inches="tight", pad_inches=0.0, dpi=150)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot a MIP as a graph.")
    parser.add_argument("--input", "-i", required=True, help="Input file in LP or MPS format.")
    parser.add_argument("--output", "-o", default="", help="Output file in PNG format.")
    parser.add_argument("--log", "-l", action="store_true", help="Use log scale.")
    parser.add_argument("--dark", "-d", action="store_true", help="Use dark theme.")
    parser.add_argument("--weighted", "-w", action="store_true", help="Weight by number of constraints.")
    args = parser.parse_args()

    # Read model
    print(f"Reading nodes and edges from {args.input}...")
    if args.input.endswith(".lp") or args.input.endswith(".lp.gz") or args.input.endswith(".lp.tar.gz"):
        model = read_lp(args.input)
    elif args.input.endswith(".mps") or args.input.endswith(".mps.gz") or args.input.endswith(".mps.tar.gz"):
        model = read_mps(args.input)
    else:
        raise ValueError(f"Unrecognized file extension: {args.input}")
    print(f"Found {len(model.variables)} variables and {len(model.constraints)} constraints.")

    # Plot model
    print("Plotting graph...")
    output_name = args.input[: args.input.rindex(".")] + ".png"
    if args.output != "":
        output_name = args.output
    plot(model, output_name, args.log, args.dark, args.weighted)


if __name__ == "__main__":
    main()

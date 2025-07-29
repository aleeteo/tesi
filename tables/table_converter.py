import os
import pandas as pd


def convert_csv_to_latex_table(csv_path, tex_path):
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"⚠️  Skipping empty or malformed file: {csv_path}")
        return

    if df.empty or df.columns.size == 0:
        print(f"⚠️  Skipping file with no data or headers: {csv_path}")
        return

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    label = f"tab:{base_name}"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(
            df.to_latex(
                index=False,
                escape=False,
                column_format="l" * len(df.columns),
                header=True,
                bold_rows=False,
                na_rep="",
                multicolumn=True,
                multicolumn_format="c",
                longtable=False,
            )
        )
        f.write("}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def main(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory, filename)
            tex_path = os.path.splitext(csv_path)[0] + ".tex"
            print(f"Converting {filename} to LaTeX...")
            convert_csv_to_latex_table(csv_path, tex_path)
    print("✅ Conversion complete.")


if __name__ == "__main__":
    main()

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Settings
# ---------------------------
base_path = r"C:\Users\pcoffice22\Documents\Meine_Dokumente\test\2025-04-30_CHO_FGFR1cHT7_SiRHTL_test\analysis\data_extraction"
ligands = ["resting", "FGF1", "FGF4", "FGF8b", "FGF9"]
file_pattern = "{ligand}_diff_coeff.txt"

# ---------------------------
# Load all files
# ---------------------------
all_data = []

for ligand in ligands:
    search_path = os.path.join(base_path, ligand, file_pattern.format(ligand=ligand))
    matches = glob.glob(search_path)

    if not matches:
        print(f"WARNING: No file found for ligand {ligand}")
        continue

    file_path = matches[0]
    df = pd.read_csv(file_path, sep="\t")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Force numeric conversion for all diffusion columns
    numeric_cols = [
        "mean D global",
        "Δmean D global",
        "mean D immobile+notype",
        "Δmean D immobile+notype",
        "mean D confined",
        "Δmean D confined",
        "mean D free",
        "Δmean D free"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to float, invalid → NaN

    df["ligand"] = ligand
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Angenommen, 'data' ist dein DataFrame
# Extract cell number as integer
data["cell_no"] = data["cell label"].str.extract(r"cell_(\d+)")[0].astype(float)

# Maske: nur cell 1 bis 5 ausschließen
mask = ~data["cell_no"].between(1, 5)
filtered_data = data[mask]

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6))

# ---------------------------
# 1. Violinplot breiter + eigene Farben
# ---------------------------
palette = sns.color_palette("Set2", n_colors=len(ligands))

sns.violinplot(
    data=filtered_data,
    x="ligand",
    y="mean D global",
    inner="quartile",         # zeigt Median + Quartile
    linewidth=1.3,
    palette=palette,
    hue="ligand",
    legend=False,
    cut=0,
    width=0.8                 # Kontrolle der Dicke (Standard 0.8)
)

# ---------------------------
# 2. Scatter/Swarm als Punktwolke (jitter)
# ---------------------------
sns.stripplot(
    data=filtered_data,
    x="ligand",
    y="mean D global",
    color="black",
    size=4,
    alpha=0.6,
    jitter=True               # sorgt für "unregelmäßiges" Muster
)

# ---------------------------
# 3. Mean als zusätzlicher Marker
# ---------------------------
group_means = filtered_data.groupby("ligand")["mean D global"].mean()

for i, ligand in enumerate(ligands):
    mean_val = group_means.get(ligand, None)
    if pd.notna(mean_val):
        plt.plot(i, mean_val, "*", color="white", markersize=14, markeredgecolor="black")

#plt.title("Global diffusion coefficient per ligand")
plt.ylabel("mean D global [µm²/s]")
plt.xlabel("Ligand")
plt.tight_layout()
plt.show()
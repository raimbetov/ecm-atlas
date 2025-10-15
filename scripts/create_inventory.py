
import pandas as pd
import os

def get_sheet_details(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        sheet_details = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            sheet_details.append((sheet_name, len(df)))
        return sheet_details
    except Exception as e:
        return [(f"Error reading file: {e}", 0)]

def get_tsv_details(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        return [('N/A', len(df))]
    except Exception as e:
        return [(f"Error reading file: {e}", 0)]

studies = {
    "Angelidis 2019": ["data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx"],
    "Ariosa-Morejon et al. 2021": [
        "data_raw/Ariosa-Morejon et al. - 2021/elife-66635-fig2-data1-v1.xlsx",
        "data_raw/Ariosa-Morejon et al. - 2021/elife-66635-fig4-data1-v1.xlsx",
        "data_raw/Ariosa-Morejon et al. - 2021/elife-66635-fig5-data1-v1.xlsx",
        "data_raw/Ariosa-Morejon et al. - 2021/elife-66635-fig6-data1-v1.xlsx",
        "data_raw/Ariosa-Morejon et al. - 2021/elife-66635-fig7-data1-v1.xlsx",
        "data_raw/Ariosa-Morejon et al. - 2021/elife-66635-fig7-data2-v1.xlsx",
    ],
    "Caldeira et al. 2017": ["data_raw/Caldeira et al. - 2017/41598_2017_11960_MOESM2_ESM.xls"],
    "Chmelova et al. 2023": ["data_raw/Chmelova et al. - 2023/Data Sheet 1.XLSX"],
    "Dipali et al. 2023": ["data_raw/Dipali et al. - 2023/Candidates.tsv"],
    "Li et al. 2021 - dermis": [
        "data_raw/Li et al. - 2021 | dermis/Table 1.xlsx",
        "data_raw/Li et al. - 2021 | dermis/Table 2.xlsx",
        "data_raw/Li et al. - 2021 | dermis/Table 3.xlsx",
        "data_raw/Li et al. - 2021 | dermis/Table 4.xlsx",
    ],
    "Li et al. 2021 - pancreas": [
        f"data_raw/Li et al. - 2021 | pancreas/41467_2021_21261_MOESM{i}_ESM.xlsx"
        for i in range(4, 12)
    ],
    "Ouni et al. 2022": [
        "data_raw/Ouni et al. - 2022/Supp Table 1.xlsx",
        "data_raw/Ouni et al. - 2022/Supp Table 2.xlsx",
        "data_raw/Ouni et al. - 2022/Supp Table 3.xlsx",
        "data_raw/Ouni et al. - 2022/Supp Table 4.xlsx",
    ],
    "Tam et al. 2020": [
        f"data_raw/Tam et al. - 2020/elife-64940-supp{i}-v3.xlsx" for i in range(1, 6)
    ],
    "Tsumagari et al. 2023": [
        f"data_raw/Tsumagari et al. - 2023/41598_2023_45570_MOESM{i}_ESM.xlsx"
        for i in range(2, 9)
    ],
}

root_path = "/Users/Kravtsovd/projects/ecm-atlas"

with open("/Users/Kravtsovd/projects/ecm-atlas/knowledge_base/00_dataset_inventory.md", "w") as f:
    f.write("| Study | File | Format | Sheet | Est. Rows | Notes |\n")
    f.write("|---|---|---|---|---|---|\n")
    for study, files in studies.items():
        for file in files:
            file_path = os.path.join(root_path, file)
            if file.endswith((".xlsx", ".XLSX", ".xls")):
                details = get_sheet_details(file_path)
                format = "Excel"
            elif file.endswith(".tsv"):
                details = get_tsv_details(file_path)
                format = "TSV"
            else:
                details = [("N/A", "N/A")]
                format = "Other"

            for sheet_name, row_count in details:
                f.write(f"| {study} | {file} | {format} | {sheet_name} | {row_count} | |\n")

print("Dataset inventory created successfully.")

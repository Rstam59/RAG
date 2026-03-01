import pandas as pd
from openpyxl import load_workbook
import numpy as np
import os  # Fayl adını path-dən ayırmaq üçün lazımdır

def smart_excel_parser(file_path):
    # Faylın təmiz adını çıxarırıq (məs: /content/data.xlsx -> data.xlsx)
    file_name = os.path.basename(file_path)
    
    print(f"🔄 Processing: {file_name}")
    
    try:
        wb = load_workbook(file_path, data_only=True)
    except Exception as e:
        return [f"Error: Could not open file - {e}"]

    all_chunks = []

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        
        # --- 1. MERGED CELLS (BİRLƏŞMİŞ XANALAR) HƏLLİ ---
        merged_ranges = list(sheet.merged_cells.ranges)
        for group in merged_ranges:
            min_col, min_row, max_col, max_row = group.bounds
            top_val = sheet.cell(row=min_row, column=min_col).value
            sheet.unmerge_cells(str(group))
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    sheet.cell(row=r, column=c).value = top_val

        # --- 2. GİZLİ SƏTİRLƏRİ ATARAQ MƏLUMATI YIĞMAQ ---
        raw_data = []
        for i in range(1, sheet.max_row + 1):
            if sheet.row_dimensions[i].hidden:
                continue
            
            row_vals = [cell.value for cell in sheet[i]]
            
            if all(v is None for v in row_vals):
                continue
            raw_data.append(row_vals)

        if not raw_data:
            continue 

        # --- 3. AĞILLI BAŞLIQ TƏYİNİ (SMART HEADER DETECTION) ---
        search_limit = min(len(raw_data), 10)
        best_header_idx = 0
        max_filled_cells = 0

        for idx in range(search_limit):
            row = raw_data[idx]
            filled_count = sum(1 for x in row if x is not None and str(x).strip() != "")
            
            if filled_count > max_filled_cells:
                max_filled_cells = filled_count
                best_header_idx = idx
        
        header_row = raw_data[best_header_idx]
        body_rows = raw_data[best_header_idx + 1:]

        clean_header = []
        for i, h in enumerate(header_row):
            if h is None or str(h).strip() == "":
                clean_header.append(f"__DROP_ME_{i}__") 
            else:
                clean_header.append(str(h).strip())

        df = pd.DataFrame(body_rows, columns=clean_header)

        # --- 4. TƏMİZLİK İŞLƏRİ (CLEANING) ---
        cols_to_drop = [c for c in df.columns if "__DROP_ME_" in c]
        df.drop(columns=cols_to_drop, inplace=True)
        
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)

        # --- 5. SERIALIZASIYA (EN - ENGLISH FORMAT) ---
        for index, row in df.iterrows():
            row_parts = []
            
            for col_name, val in row.items():
                if pd.isna(val) or str(val).strip() == "":
                    continue
                
                if isinstance(val, pd.Timestamp):
                    val = val.strftime('%Y-%m-%d')
                
                if isinstance(val, float) and val.is_integer():
                    val = int(val)

                row_parts.append(f"{col_name}: {val}")
            
            if len(row_parts) >= 1:
                # Hesablanmış real sətir nömrəsi
                real_row_num = index + best_header_idx + 2
                
                # YENİ FORMAT:
                # 1. File Name (Birinci gəlir)
                # 2. Type: Table Row (Cədvəl olduğunu bildirir)
                # 3. Sheet & Row Info
                chunk = f"File: {file_name} | Type: Table Row | Sheet: {sheet_name} | Row Number: {real_row_num}\n"
                chunk += ", ".join(row_parts)
                
                all_chunks.append(chunk)

    return all_chunks
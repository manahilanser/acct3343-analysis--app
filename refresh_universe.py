from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "universe.csv"
CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "master/data/constituents.csv"
)
EXCLUDED_SECTORS = {"Financials", "Financial Services"}
MAX_WORKERS = 3


def _get_value(frame, names, column):
    if frame is None or column not in frame.columns:
        return None
    for name in names:
        if name in frame.index:
            value = frame.at[name, column]
            if pd.notna(value):
                return float(value)
    return None


def _statement_value(frames, names, period_index):
    for frame in frames:
        if frame is not None and len(frame.columns) > period_index:
            value = _get_value(frame, names, frame.columns[period_index])
            if value is not None:
                return value
    return None


def _download_company(symbol, metadata, existing, retrieved_at):
    yahoo_symbol = symbol.replace(".", "-")
    last_error = None
    for attempt in range(3):
        try:
            company = yf.Ticker(yahoo_symbol)
            balance = company.balance_sheet
            income = company.income_stmt
            cashflow = company.cashflow
            info = company.get_info()
            market_cap = info.get("marketCap")
            if market_cap is None:
                market_cap = company.fast_info.get("market_cap")

            old = existing.get(symbol, {})
            row = {
                "ticker": symbol,
                "name": info.get("longName") or info.get("shortName") or metadata.get("Security") or old.get("name"),
                "sector": metadata.get("GICS Sector") or old.get("sector"),
                "industry": metadata.get("GICS Sub-Industry") or old.get("industry"),
                "gics_sector": metadata.get("GICS Sector") or old.get("gics_sector"),
                "gics_sub_industry": metadata.get("GICS Sub-Industry") or old.get("gics_sub_industry"),
                "market_cap": float(market_cap) if market_cap is not None else None,
                "price": info.get("currentPrice") or info.get("regularMarketPrice") or old.get("price"),
                "data_as_of": retrieved_at,
            }
            fields = {
                "total_assets": ([balance], ["Total Assets"], 0),
                "total_assets_prev": ([balance], ["Total Assets"], 1),
                "total_liabilities": ([balance], ["Total Liabilities Net Minority Interest", "Total Liabilities"], 0),
                "total_liabilities_prev": ([balance], ["Total Liabilities Net Minority Interest", "Total Liabilities"], 1),
                "current_assets": ([balance], ["Current Assets"], 0),
                "current_assets_prev": ([balance], ["Current Assets"], 1),
                "current_liabilities": ([balance], ["Current Liabilities"], 0),
                "current_liabilities_prev": ([balance], ["Current Liabilities"], 1),
                "cash": ([balance], ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"], 0),
                "cash_prev": ([balance], ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"], 1),
                "receivables": ([balance], ["Receivables", "Accounts Receivable"], 0),
                "receivables_prev": ([balance], ["Receivables", "Accounts Receivable"], 1),
                "net_ppe": ([balance], ["Net PPE"], 0),
                "net_ppe_prev": ([balance], ["Net PPE"], 1),
                "retained_earnings": ([balance], ["Retained Earnings"], 0),
                "retained_earnings_prev": ([balance], ["Retained Earnings"], 1),
                "equity": ([balance], ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"], 0),
                "equity_prev": ([balance], ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"], 1),
                "total_debt": ([balance], ["Total Debt"], 0),
                "total_debt_prev": ([balance], ["Total Debt"], 1),
                "revenue": ([income], ["Total Revenue", "Operating Revenue"], 0),
                "revenue_prev": ([income], ["Total Revenue", "Operating Revenue"], 1),
                "gross_profit": ([income], ["Gross Profit"], 0),
                "gross_profit_prev": ([income], ["Gross Profit"], 1),
                "ebit": ([income], ["EBIT"], 0),
                "ebit_prev": ([income], ["EBIT"], 1),
                "net_income": ([income], ["Net Income", "Net Income Common Stockholders"], 0),
                "net_income_prev": ([income], ["Net Income", "Net Income Common Stockholders"], 1),
                "sga": ([income], ["Selling General And Administration"], 0),
                "sga_prev": ([income], ["Selling General And Administration"], 1),
                "depreciation": ([cashflow, income], ["Depreciation And Amortization", "Reconciled Depreciation"], 0),
                "depreciation_prev": ([cashflow, income], ["Depreciation And Amortization", "Reconciled Depreciation"], 1),
                "operating_cash_flow": ([cashflow], ["Operating Cash Flow"], 0),
                "operating_cash_flow_prev": ([cashflow], ["Operating Cash Flow"], 1),
            }
            for field, (frames, names, period_index) in fields.items():
                row[field] = _statement_value(frames, names, period_index)
            return row, None
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return None, f"{symbol}: {last_error}"


def main():
    current = pd.read_csv(UNIVERSE_PATH)
    current = current[~current["sector"].isin(EXCLUDED_SECTORS)]
    old_rows = current.set_index("ticker").to_dict(orient="index")

    constituents = pd.read_csv(CONSTITUENTS_URL).fillna("")
    constituents = constituents[~constituents["GICS Sector"].isin(EXCLUDED_SECTORS)]
    metadata = {
        row["Symbol"]: row
        for row in constituents.to_dict(orient="records")
    }
    symbols = sorted(set(old_rows) | set(metadata))
    retrieved_at = datetime.now(timezone.utc).date().isoformat()
    print(f"Refreshing {len(symbols)} nonfinancial companies...")

    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _download_company,
                symbol,
                metadata.get(symbol, {}),
                old_rows,
                retrieved_at,
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row, error = future.result()
            if row and row.get("market_cap") and row.get("total_assets"):
                rows.append(row)
            elif error:
                errors.append(error)
            if completed % 50 == 0 or completed == len(futures):
                print(f"Fetched {completed}/{len(futures)}; usable rows: {len(rows)}")

    if len(rows) < 500:
        raise RuntimeError(
            f"Only {len(rows)} companies have market cap and total assets; "
            "leaving universe.csv unchanged."
        )

    result = pd.DataFrame(rows).sort_values("ticker")
    temporary_path = UNIVERSE_PATH.with_suffix(".csv.tmp")
    result.to_csv(temporary_path, index=False)
    temporary_path.replace(UNIVERSE_PATH)
    print(f"Saved {len(result)} companies to {UNIVERSE_PATH.name}.")
    print(f"Data retrieval date: {retrieved_at}")
    print(f"Companies skipped: {len(symbols) - len(rows)}")
    if errors:
        print("Sample errors:")
        for error in errors[:10]:
            print(f"- {error}")


if __name__ == "__main__":
    main()

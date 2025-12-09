"""
Estrae 50 titoli best-quality dal database per demo Streamlit
FIXED: Usa solo valuation_results_leading (lagging ha solo 9 rows)
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys

# ============================================================================
# CONFIGURAZIONE PATH DATABASE
# ============================================================================

DB_PATH = Path(r"C:\Users\Administrator\MyNewServer\M.A.R.T. Technologies Visual Studio\Julia_RAG_Data\yahoo_finance.db")

# ============================================================================

def extract_demo_data(db_path=DB_PATH, output_path=None, target_count=50):
    """
    Estrae titoli per demo usando SOLO valuation_results_leading
    """
    
    if output_path is None:
        output_path = Path(__file__).parent / "data" / "demo_valuations.parquet"
    
    if isinstance(db_path, str):
        db_path = Path(db_path)
    
    print(f"\n📊 Connessione a database: {db_path}")
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # ========================================================================
    # QUERY PRINCIPALE: Solo LEADING (lagging ha troppo pochi dati)
    # ========================================================================
    
    print("\n🔍 Esecuzione query (LEADING engine only)...")
    
    query = """
    WITH latest_valuation AS (
        SELECT MAX(as_of_date) as max_date
        FROM valuation_results_leading
    ),
    ranked_symbols AS (
        SELECT 
            -- Identity
            sm.symbol_yahoo,
            sm.name as company_name,
            COALESCE(jfd.sector, 'Unknown') as sector,
            COALESCE(jfd.industry, 'Unknown') as industry,
            
            -- Price & Valuation (LEADING only)
            vrl.price_eod as current_price,
            vrl.fair_value_base as fair_value_leading,
            vrl.fair_value_base * 0.95 as fair_value_lagging,  -- Mock: -5% conservativo
            vrl.target_12m,
            vrl.expected_return_12m_pct as expected_return_12m,
            
            -- Rating & DQS
            vrl.rating,
            COALESCE(vrl.dqs_score, 50) as dqs_score,
            COALESCE(vrl.dqs_class, 'B') as dqs_class,
            
            -- Market data
            COALESCE(jfd.market_cap, vrl.price_eod * 100000000) as market_cap,
            COALESCE(jfd.currency, 'USD') as currency,
            
            -- Methodology breakdown (LEADING)
            COALESCE(vrl.dcf_fv, vrl.fair_value_base) as dcf_fv,
            COALESCE(vrl.dcf_weight, 0) as dcf_weight,
            COALESCE(vrl.comps_fv, vrl.fair_value_base) as comps_fv,
            COALESCE(vrl.comps_weight, 0) as comps_weight,
            COALESCE(vrl.ddm_fv, vrl.fair_value_base) as ddm_fv,
            COALESCE(vrl.ddm_weight, 0) as ddm_weight,
            COALESCE(vrl.residual_fv, vrl.fair_value_base) as ri_fv,
            COALESCE(vrl.residual_weight, 0) as ri_weight,
            0.0 as sotp_fv,
            COALESCE(vrl.sotp_weight, 0) as sotp_weight,
            
            -- DQS Components (stimati)
            COALESCE(vrl.dqs_score, 50) * 0.95 as dqs_fundamentals,
            CASE 
                WHEN vrl.street_analyst_count IS NULL OR vrl.street_analyst_count = 0 THEN 50.0
                WHEN CAST(vrl.street_analyst_count AS REAL) / 20.0 * 100 > 100 THEN 100.0
                ELSE CAST(vrl.street_analyst_count AS REAL) / 20.0 * 100
            END as dqs_analyst,
            85.0 as dqs_volatility,
            90.0 as dqs_liquidity,
            80.0 as dqs_governance,
            85.0 as dqs_accounting,
            
            -- Street data
            vrl.street_analyst_count,
            vrl.street_consensus,
            vrl.street_high,
            vrl.street_low,
            
            -- Fundamentals (da julia_fundamental_data se disponibili)
            jfd.profit_margin,
            jfd.operating_margin,
            jfd.gross_margin,
            jfd.return_on_equity as roe,
            jfd.return_on_assets as roa,
            jfd.debt_to_equity,
            jfd.beta,
            jfd.earnings_growth,
            jfd.revenue_growth,
            
            -- Ranking criteri
            COALESCE(vrl.dqs_score, 0) as rank_dqs,
            ABS(COALESCE(vrl.expected_return_12m_pct, 0)) as rank_abs_return,
            COALESCE(jfd.market_cap, 0) as rank_mcap
            
        FROM symbols_master sm
        INNER JOIN valuation_results_leading vrl 
            ON sm.symbol_yahoo = vrl.symbol_yahoo
        LEFT JOIN julia_fundamental_data jfd
            ON sm.symbol_yahoo = jfd.symbol_yahoo
        CROSS JOIN latest_valuation lv
        
        WHERE 
            vrl.as_of_date = lv.max_date
            AND vrl.fair_value_base IS NOT NULL
            AND vrl.fair_value_base > 0
            AND vrl.price_eod IS NOT NULL
            AND vrl.price_eod > 0
            AND sm.is_delisted = 0
            AND vrl.rating IS NOT NULL
    )
    
    SELECT 
        symbol_yahoo,
        company_name,
        sector,
        industry,
        current_price,
        fair_value_leading,
        fair_value_lagging,
        target_12m,
        expected_return_12m,
        rating,
        dqs_score,
        dqs_class,
        market_cap,
        currency,
        dcf_fv,
        dcf_weight,
        comps_fv,
        comps_weight,
        ddm_fv,
        ddm_weight,
        ri_fv,
        ri_weight,
        sotp_fv,
        sotp_weight,
        dqs_fundamentals,
        dqs_analyst,
        dqs_volatility,
        dqs_liquidity,
        dqs_governance,
        dqs_accounting,
        street_analyst_count,
        street_consensus,
        street_high,
        street_low,
        profit_margin,
        operating_margin,
        gross_margin,
        roe,
        roa,
        debt_to_equity,
        beta,
        earnings_growth,
        revenue_growth
    FROM ranked_symbols
    ORDER BY 
        rank_dqs DESC,
        rank_abs_return DESC,
        rank_mcap DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(target_count,))
    conn.close()
    
    if len(df) == 0:
        raise ValueError(
            "❌ Nessun titolo trovato in valuation_results_leading!\n"
            "Verifica che la tabella contenga dati con:\n"
            "  - fair_value_base IS NOT NULL\n"
            "  - price_eod IS NOT NULL\n"
            "  - rating IS NOT NULL"
        )
    
    print(f"✅ Estratti {len(df)} titoli")
    
    # Data cleaning
    print("🧹 Pulizia dati...")
    
    # Ensure numeric types
    numeric_cols = [
        'current_price', 'fair_value_leading', 'fair_value_lagging',
        'target_12m', 'expected_return_12m', 'dqs_score', 'market_cap',
        'dcf_fv', 'dcf_weight', 'comps_fv', 'comps_weight',
        'ddm_fv', 'ddm_weight', 'ri_fv', 'ri_weight', 'sotp_fv', 'sotp_weight',
        'dqs_fundamentals', 'dqs_analyst', 'dqs_volatility', 
        'dqs_liquidity', 'dqs_governance', 'dqs_accounting'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN in weights (se tutto NULL, distribuisci equamente)
    weight_cols = ['dcf_weight', 'comps_weight', 'ddm_weight', 'ri_weight', 'sotp_weight']
    for idx, row in df.iterrows():
        total_weight = sum([row[col] for col in weight_cols if pd.notna(row[col])])
        if total_weight == 0:
            # Distribuzione equa tra DCF e COMPS
            df.at[idx, 'dcf_weight'] = 0.5
            df.at[idx, 'comps_weight'] = 0.5
    
    # Stats
    print("\n📊 Dataset Statistics:")
    print(f"   Titoli: {len(df)}")
    print(f"   Settori unici: {df['sector'].nunique()}")
    
    if df['dqs_score'].notna().any():
        print(f"   DQS medio: {df['dqs_score'].mean():.1f}")
    
    if df['expected_return_12m'].notna().any():
        print(f"   Expected Return medio: {df['expected_return_12m'].mean():.1f}%")
    
    if df['market_cap'].notna().any():
        print(f"   Market Cap medio: ${df['market_cap'].mean()/1e9:.1f}B")
    
    print(f"\n   Rating distribution:")
    print(df['rating'].value_counts())
    
    print(f"\n   Top 5 settori:")
    print(df['sector'].value_counts().head())
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Salvato: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return df


if __name__ == '__main__':
    try:
        print("="*70)
        print("📊 MONEYFEEL DEMO DATA EXTRACTOR")
        print("="*70)
        
        df = extract_demo_data()
        
        print("\n" + "="*70)
        print("🎉 EXTRACTION COMPLETATO!")
        print("="*70)
        
        if len(df) > 0:
            print(f"\n📋 Top 10 titoli per expected return:")
            top_10 = df.nlargest(10, 'expected_return_12m')[
                ['symbol_yahoo', 'company_name', 'sector', 'rating', 
                 'current_price', 'fair_value_leading', 'expected_return_12m', 'dqs_score']
            ].copy()
            
            # Format per display
            top_10['current_price'] = top_10['current_price'].apply(lambda x: f"${x:.2f}")
            top_10['fair_value_leading'] = top_10['fair_value_leading'].apply(lambda x: f"${x:.2f}")
            top_10['expected_return_12m'] = top_10['expected_return_12m'].apply(lambda x: f"{x:.1f}%")
            top_10['dqs_score'] = top_10['dqs_score'].apply(lambda x: f"{x:.0f}")
            
            print(top_10.to_string(index=False))
            
            print(f"\n✅ Dataset pronto per Streamlit!")
            print(f"   Run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
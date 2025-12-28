"""
Populate database with test data to verify the system works
"""

from db.core import SessionLocal
from db.models import PerfMetric
from datetime import datetime
import random

def populate_test_data():
    """Add sample stock data to database"""
    db = SessionLocal()
    
    try:
        # Clear existing data
        db.query(PerfMetric).delete()
        db.commit()
        print("✅ Cleared existing data")
        
        # Sample stocks
        stocks = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD',
            'NFLX', 'INTC', 'CSCO', 'ADBE', 'CRM', 'AVGO', 'QCOM', 'TXN',
            'ORCL', 'IBM', 'AMAT', 'LRCX', 'KLAC', 'SNPS', 'MRVL', 'PYPL',
            'BABA', 'NIO', 'XPEV', 'LI', 'BIDU', 'JD', 'PDD', 'COIN',
            'SQ', 'ROKU', 'SPOT', 'ZM', 'DOCU', 'CRWD', 'OKTA', 'SNOW',
            'PLTR', 'DKNG', 'PTON', 'MCHP', 'VIPS', 'TME', 'YMM', 'WB',
            'DIDI', 'BILI'
        ]
        
        strategies = ['scalp', 'swing', 'longterm']
        
        records_added = 0
        
        for strategy in strategies:
            print(f"\nProcessing {strategy} strategy...")
            
            for i, symbol in enumerate(stocks):
                # Generate realistic performance metrics
                metric_x = random.uniform(-20, 30)  # Short-term return %
                metric_y = random.uniform(-15, 25)  # Long-term return %
                
                # Calculate z-scores
                z_x = (metric_x - 5) / 10  # Simplified z-score
                z_y = (metric_y - 5) / 10
                
                # Mark as outlier if z-score > 2 or < -2
                is_outlier = abs(z_x) > 2.0 or abs(z_y) > 2.0
                
                # Create record
                record = PerfMetric(
                    symbol=symbol,
                    strategy=strategy,
                    metric_x=metric_x,
                    metric_y=metric_y,
                    z_x=z_x,
                    z_y=z_y,
                    is_outlier=is_outlier,
                    inserted=datetime.now()
                )
                
                db.add(record)
                records_added += 1
                
                if is_outlier:
                    print(f"  📍 {symbol}: outlier (z_x={z_x:.2f}, z_y={z_y:.2f})")
            
            db.commit()
            print(f"✅ Added {len(stocks)} records for {strategy}")
        
        # Summary
        total = db.query(PerfMetric).count()
        outliers = db.query(PerfMetric).filter(PerfMetric.is_outlier == True).count()
        
        print(f"\n" + "="*50)
        print(f"✅ Database populated successfully!")
        print(f"📊 Total records: {total}")
        print(f"🎯 Total outliers: {outliers}")
        print(f"📈 Normal stocks: {total - outliers}")
        print("="*50)
        
        # Show breakdown by strategy
        print("\nBreakdown by strategy:")
        for strategy in strategies:
            count = db.query(PerfMetric).filter(PerfMetric.strategy == strategy).count()
            outlier_count = db.query(PerfMetric).filter(
                PerfMetric.strategy == strategy,
                PerfMetric.is_outlier == True
            ).count()
            print(f"  {strategy}: {count} stocks ({outlier_count} outliers)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Populating BILLIONS database with test data...")
    print("="*50)
    populate_test_data()
    print("\n✅ Done! You can now test the Outliers page.")
    print("   Go to: http://localhost:3000/outliers")


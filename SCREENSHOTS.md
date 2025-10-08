# 📸 Visual Assets & Screenshots Guide

This document provides guidance on creating and organizing visual assets for the BILLIONS project.

## 🎨 Available Assets

### Logos & Branding

Located in `funda/assets/`:

1. **Main Logo**
   - File: `logo.png`
   - Usage: README header, documentation
   - Recommended size: 200x200px for README

2. **Motivational Logo**
   - File: `nanakorobi_yaoki.png`
   - Translation: "七転び八起き" (Fall seven times, stand up eight)
   - Usage: README footer, about section
   - Recommended size: 150x150px

### Custom Fonts

Available fonts for UI customization:

- **DePixel Series** (`depixel/`)
  - Modern, pixel-art style
  - Multiple weights available
  - Format: .otf, .ttf, .woff

- **Enhanced Dot Digital-7** 
  - File: `enhanced_dot_digital-7.ttf`
  - Perfect for numerical displays
  - Great for stock prices and metrics

- **Minecraft Font**
  - File: `Minecraft.ttf`
  - Fun, blocky style
  - Optional for playful elements

### Font Configuration

The dashboard uses custom fonts via `funda/assets/custom-font.css`:
```css
@font-face {
    font-family: 'CustomFont';
    src: url('path/to/font.ttf') format('truetype');
}
```

---

## 📊 Screenshots to Create

### 1. Dashboard Overview

**Recommended Composition:**
```
┌─────────────────────────────────────────────┐
│  BILLIONS ML PREDICTION SYSTEM              │
│  [Input Box: TSLA] [🚀 Run Prediction]      │
├─────────────────────────────────────────────┤
│                                             │
│  📈 Candlestick Chart                       │
│     (with Bollinger Bands overlay)          │
│                                             │
├─────────────────────────────────────────────┤
│  30-Day Predictions Table                   │
│  | Date | Predicted | Confidence |          │
└─────────────────────────────────────────────┘
```

**Filename:** `screenshots/dashboard_overview.png`

**Capture Settings:**
- Resolution: 1920x1080 or higher
- Browser: Chrome (for consistency)
- Zoom: 100%
- Ticker: Use popular stock (TSLA, NVDA, AAPL)

### 2. Technical Analysis View

**Focus On:**
- Multiple indicator overlays (RSI, MACD, Bollinger Bands)
- Volume chart
- Clear annotations

**Filename:** `screenshots/technical_analysis.png`

### 3. Outlier Detection

**Show:**
- Scatter plot with Z-scores
- Highlighted outlier stocks
- Performance metrics

**Filename:** `screenshots/outlier_detection.png`

### 4. Prediction Results

**Capture:**
- 30-day forecast table
- Confidence scores
- Current vs. predicted prices

**Filename:** `screenshots/predictions.png`

### 5. Performance Metrics

**Display:**
- Win rate
- Accuracy metrics
- Sharpe ratio
- Drawdown chart

**Filename:** `screenshots/performance_metrics.png`

---

## 🎥 GIF Animations (Optional)

Create short GIFs showing:

### 1. Quick Prediction Demo
```
1. Enter ticker → 2. Click button → 3. View results
Duration: 5-10 seconds
```

**Filename:** `screenshots/demo.gif`

### 2. Outlier Discovery
```
1. Open outlier tab → 2. Filter by strategy → 3. Explore results
Duration: 5-10 seconds
```

**Filename:** `screenshots/outlier_demo.gif`

### Tools for Creating GIFs:
- **ScreenToGif** (Windows)
- **LICEcap** (Mac/Windows)
- **Peek** (Linux)

---

## 📐 Diagram Assets

### Architecture Diagram

Create a visual representation of `SYSTEM_FLOWCHART.md`:

**Suggested Tools:**
- **Draw.io** (free, web-based)
- **Lucidchart**
- **Mermaid** (code-based diagrams)

**Filename:** `screenshots/architecture.png`

**Elements to Include:**
```
┌──────────┐     ┌──────────┐     ┌──────────┐
│   User   │────▶│Dashboard │────▶│  LSTM    │
└──────────┘     └──────────┘     └──────────┘
                       │
                       ▼
                 ┌──────────┐
                 │ Database │
                 └──────────┘
```

### Data Flow Diagram

**Show:**
- Data sources (Yahoo Finance, Alpha Vantage)
- Processing pipeline
- Prediction output

**Filename:** `screenshots/data_flow.png`

---

## 🎨 Color Palette

For consistent branding across visuals:

### Primary Colors
```
Dark Blue:   #1E3A8A (Headers, primary elements)
Light Blue:  #3B82F6 (Accents, links)
Green:       #10B981 (Positive predictions, gains)
Red:         #EF4444 (Negative predictions, losses)
```

### Background Colors
```
Dark Mode:   #1F2937 (Main background)
Light Mode:  #F9FAFB (Main background)
Cards:       #FFFFFF (Light) / #374151 (Dark)
```

### Chart Colors
```
Bullish Candle:  #10B981 (Green)
Bearish Candle:  #EF4444 (Red)
Volume:          #6B7280 (Gray)
MA Lines:        #3B82F6, #8B5CF6, #EC4899 (Blue, Purple, Pink)
```

---

## 📝 Screenshot Guidelines

### Do's ✅

- Use realistic, well-known stock tickers (TSLA, NVDA, AAPL)
- Show meaningful data (avoid all zeros or NaN)
- Capture full features (don't crop important UI)
- Use consistent window size across screenshots
- Show successful predictions/results
- Include timestamps to show real-time capability
- Use high resolution (1920x1080 minimum)

### Don'ts ❌

- Don't include personal API keys
- Don't show error messages (unless for troubleshooting docs)
- Don't use obscure penny stocks
- Don't show unrealistic gains (pump & dump stocks)
- Don't include personal information
- Don't use low-resolution images
- Don't mix light/dark themes across screenshots

---

## 🖼️ Image Optimization

### Before Adding to Repository

1. **Compress Images**
   ```bash
   # Using ImageOptim (Mac)
   # Using TinyPNG (Web)
   # Using pngquant (CLI)
   pngquant --quality=65-80 screenshot.png
   ```

2. **Recommended Formats**
   - **Screenshots**: PNG (for sharp UI elements)
   - **Photos/Logos**: JPG (smaller file size)
   - **Animations**: GIF or WebP
   - **Diagrams**: SVG (scalable, small size)

3. **File Size Limits**
   - Individual images: < 1MB
   - GIFs: < 5MB
   - Total screenshots folder: < 20MB

---

## 📁 Folder Structure

Organize visual assets:

```
Billions/
├── screenshots/
│   ├── dashboard_overview.png
│   ├── technical_analysis.png
│   ├── outlier_detection.png
│   ├── predictions.png
│   ├── performance_metrics.png
│   ├── demo.gif
│   ├── architecture.png
│   └── data_flow.png
│
├── funda/assets/
│   ├── logo.png
│   ├── nanakorobi_yaoki.png
│   ├── custom-font.css
│   ├── depixel/
│   ├── enhanced_dot_digital_7/
│   └── minecraft/
│
└── docs/
    └── images/
        └── (additional documentation images)
```

---

## 🚀 Adding Screenshots to README

### Example Markdown

```markdown
## 📊 Dashboard Preview

<div align="center">
  <img src="screenshots/dashboard_overview.png" alt="Dashboard Overview" width="800"/>
  <p><i>Main dashboard with LSTM predictions</i></p>
</div>

## 🎯 Outlier Detection

<div align="center">
  <img src="screenshots/outlier_detection.png" alt="Outlier Detection" width="800"/>
  <p><i>Identifying high-potential stocks</i></p>
</div>
```

### GIF Demo

```markdown
## 🎥 Quick Demo

<div align="center">
  <img src="screenshots/demo.gif" alt="Quick Demo" width="600"/>
  <p><i>Running a prediction in seconds</i></p>
</div>
```

---

## 🎬 Video Tutorials (Future)

Consider creating YouTube tutorials:

1. **Installation & Setup** (5 min)
2. **First Prediction** (3 min)
3. **Understanding Technical Indicators** (10 min)
4. **Outlier Detection Strategy** (8 min)
5. **Training Custom Models** (12 min)

**Embed in README:**
```markdown
[![BILLIONS Tutorial](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
```

---

## 🎨 Design Resources

### Icon Sets (Free)
- [Font Awesome](https://fontawesome.com/)
- [Heroicons](https://heroicons.com/)
- [Feather Icons](https://feathericons.com/)

### Color Tools
- [Coolors](https://coolors.co/) - Color palette generator
- [ColorHunt](https://colorhunt.co/) - Curated palettes
- [Adobe Color](https://color.adobe.com/) - Color wheel

### Screenshot Tools
- **Windows**: Snipping Tool, ShareX, Greenshot
- **Mac**: Screenshot (⌘+Shift+4), CleanShot X
- **Linux**: Flameshot, Shutter
- **Cross-platform**: OBS Studio (for videos)

---

## ✅ Checklist for Release

Before releasing to GitHub:

- [ ] Logo added to README header
- [ ] At least 3 core screenshots captured
- [ ] Architecture diagram created
- [ ] Images compressed and optimized
- [ ] All screenshots show realistic data
- [ ] No sensitive information visible
- [ ] Consistent theme (light/dark) across images
- [ ] GIF demo created (optional)
- [ ] Alt text added to all images
- [ ] Images referenced correctly in README

---

## 💡 Tips for Great Screenshots

1. **Timing**: Capture during market hours for realistic data
2. **Data**: Use well-known stocks with interesting patterns
3. **Cleanliness**: Close unnecessary browser tabs
4. **Focus**: Highlight the feature you're demonstrating
5. **Annotations**: Add arrows or highlights for key features
6. **Consistency**: Use the same ticker across related screenshots

---

**Ready to make BILLIONS look amazing!** 🎨

[Back to README](README.md)


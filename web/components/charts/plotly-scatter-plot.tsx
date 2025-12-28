'use client';

import dynamic from 'next/dynamic';
import { Badge } from '@/components/ui/badge';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface DataPoint {
  symbol: string;
  x: number;
  y: number;
  isOutlier: boolean;
}

interface PlotlyScatterPlotProps {
  data: DataPoint[];
  xLabel?: string;
  yLabel?: string;
  title?: string;
}

export function PlotlyScatterPlot({ 
  data, 
  xLabel = 'X Metric', 
  yLabel = 'Y Metric',
  title = 'Outlier Analysis'
}: PlotlyScatterPlotProps) {
  
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 border rounded-lg bg-muted/10">
        <div className="text-center">
          <div className="text-muted-foreground text-lg mb-2">📊</div>
          <div className="text-muted-foreground">No data available</div>
        </div>
      </div>
    );
  }

  // Separate normal and outlier points
  const normalPoints = data.filter(point => !point.isOutlier);
  const outlierPoints = data.filter(point => point.isOutlier);

  // Create traces for normal and outlier points
  const traces = [];

  // Normal points trace
  if (normalPoints.length > 0) {
    traces.push({
      x: normalPoints.map(p => p.x),
      y: normalPoints.map(p => p.y),
      text: normalPoints.map(p => p.symbol),
      mode: 'markers+text',
      type: 'scatter',
      name: 'Normal',
      marker: {
        color: '#3b82f6', // Blue
        size: 8,
        opacity: 0.7
      },
      textposition: 'top center',
      textfont: {
        size: 10,
        color: '#374151'
      },
      hovertemplate: '<b>%{text}</b><br>' +
                    `${xLabel}: %{x:.2f}%<br>` +
                    `${yLabel}: %{y:.2f}%<br>` +
                    '<extra></extra>'
    });
  }

  // Outlier points trace
  if (outlierPoints.length > 0) {
    traces.push({
      x: outlierPoints.map(p => p.x),
      y: outlierPoints.map(p => p.y),
      text: outlierPoints.map(p => p.symbol),
      mode: 'markers+text',
      type: 'scatter',
      name: 'Outlier',
      marker: {
        color: '#ef4444', // Red
        size: 12,
        opacity: 0.9,
        line: {
          color: '#dc2626',
          width: 2
        }
      },
      textposition: 'top center',
      textfont: {
        size: 11,
        color: '#374151',
        family: 'Arial, sans-serif'
      },
      hovertemplate: '<b>%{text}</b><br>' +
                    `${xLabel}: %{x:.2f}%<br>` +
                    `${yLabel}: %{y:.2f}%<br>` +
                    '<extra></extra>'
    });
  }

  const layout = {
    title: {
      text: title,
      font: { size: 18, family: 'Arial, sans-serif' }
    },
    xaxis: {
      title: {
        text: xLabel,
        font: { size: 14, family: 'Arial, sans-serif' }
      },
      showgrid: true,
      gridcolor: '#e5e7eb',
      zeroline: true,
      zerolinecolor: '#000',
      zerolinewidth: 1
    },
    yaxis: {
      title: {
        text: yLabel,
        font: { size: 14, family: 'Arial, sans-serif' }
      },
      showgrid: true,
      gridcolor: '#e5e7eb',
      zeroline: true,
      zerolinecolor: '#000',
      zerolinewidth: 1
    },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    font: {
      family: 'Arial, sans-serif',
      color: '#374151'
    },
    margin: {
      l: 80,
      r: 50,
      t: 80,
      b: 80
    },
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(255,255,255,0.8)',
      bordercolor: '#e5e7eb',
      borderwidth: 1
    },
    hovermode: 'closest',
    dragmode: 'zoom' as const, // Enable zoom and pan
    modebar: {
      orientation: 'v',
      bgcolor: 'rgba(255,255,255,0.8)',
      color: '#374151',
      activecolor: '#3b82f6'
    }
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    responsive: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'outlier_analysis',
      height: 800,
      width: 1200,
      scale: 2
    }
  };

  return (
    <div className="w-full">
      {/* Stats */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex gap-2">
          <Badge variant="outline" className="text-blue-600 border-blue-600">
            Normal: {normalPoints.length}
          </Badge>
          <Badge variant="destructive">
            Outliers: {outlierPoints.length}
          </Badge>
        </div>
        <div className="text-sm text-muted-foreground">
          💡 Mouse wheel: zoom | Drag: pan | Double-click: reset
        </div>
      </div>

      {/* Plotly Chart */}
      <div className="border rounded-lg bg-white overflow-hidden">
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '600px' }}
          useResizeHandler={true}
        />
      </div>
    </div>
  );
}

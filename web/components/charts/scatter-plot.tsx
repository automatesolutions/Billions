'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';

interface DataPoint {
  symbol: string;
  x: number;
  y: number;
  isOutlier: boolean;
}

interface ScatterPlotProps {
  data: DataPoint[];
  xLabel?: string;
  yLabel?: string;
}

export function ScatterPlot({ data, xLabel = 'X Metric', yLabel = 'Y Metric' }: ScatterPlotProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 }); // Will be set based on data
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const svgRef = useRef<SVGSVGElement>(null);

  if (!data || data.length === 0) {
    return <div className="text-muted-foreground text-sm">No data available</div>;
  }

  console.log('ScatterPlot data:', data);

  const xValues = data.map(d => d.x);
  const yValues = data.map(d => d.y);
  
  const xMax = Math.max(...xValues);
  const xMin = Math.min(...xValues);
  const yMax = Math.max(...yValues);
  const yMin = Math.min(...yValues);
  
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  // Calculate data bounds with padding
  const padding = 5; // 5% padding around data
  const dataMinX = xMin - (xRange * padding / 100);
  const dataMaxX = xMax + (xRange * padding / 100);
  const dataMinY = yMin - (yRange * padding / 100);
  const dataMaxY = yMax + (yRange * padding / 100);
  
  // For zoom out, always show the full data range
  const dataWidth = dataMaxX - dataMinX;
  const dataHeight = dataMaxY - dataMinY;
  const dataCenterX = (dataMinX + dataMaxX) / 2;
  const dataCenterY = (dataMinY + dataMaxY) / 2;
  
  // Calculate viewBox - when zoomed out, show full data
  let viewBoxSize, viewBoxX, viewBoxY;
  
  if (zoom <= 1) {
    // Show full data when zoomed out
    viewBoxSize = Math.max(dataWidth, dataHeight);
    viewBoxX = dataMinX - (viewBoxSize - dataWidth) / 2;
    viewBoxY = dataMinY - (viewBoxSize - dataHeight) / 2;
  } else {
    // When zoomed in, allow panning
    viewBoxSize = Math.max(dataWidth, dataHeight) / zoom;
    viewBoxX = Math.max(dataMinX, Math.min(dataMaxX - viewBoxSize, pan.x - viewBoxSize / 2));
    viewBoxY = Math.max(dataMinY, Math.min(dataMaxY - viewBoxSize, pan.y - viewBoxSize / 2));
  }

  // Zoom handlers
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.2, Math.min(10, zoom * zoomFactor));
    setZoom(newZoom);
  }, [zoom]);

  // Pan handlers
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging) return;
    const deltaX = (e.clientX - dragStart.x) / 10; // Scale down movement
    const deltaY = (e.clientY - dragStart.y) / 10;
    setPan(prev => ({
      x: Math.max(dataCenterX - dataWidth/2, Math.min(dataCenterX + dataWidth/2, prev.x - deltaX)),
      y: Math.max(dataCenterY - dataHeight/2, Math.min(dataCenterY + dataHeight/2, prev.y + deltaY))
    }));
  }, [isDragging, dragStart, dataCenterX, dataCenterY, dataWidth, dataHeight]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Reset zoom and pan
  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: dataCenterX, y: dataCenterY });
  }, [dataCenterX, dataCenterY]);

  // Set initial pan position when data loads
  useEffect(() => {
    if (data && data.length > 0) {
      setPan({ x: dataCenterX, y: dataCenterY });
    }
  }, [dataCenterX, dataCenterY]);

  return (
    <div className="space-y-2">
      {/* Controls */}
      <div className="flex justify-between items-center">
        <div className="flex gap-2">
          <button
            onClick={() => setZoom(prev => Math.max(0.1, prev - 0.2))}
            className="px-3 py-1 text-xs border rounded hover:bg-muted"
          >
            Zoom Out
          </button>
          <button
            onClick={() => setZoom(prev => Math.min(5, prev + 0.2))}
            className="px-3 py-1 text-xs border rounded hover:bg-muted"
          >
            Zoom In
          </button>
          <button
            onClick={resetView}
            className="px-3 py-1 text-xs border rounded hover:bg-muted"
          >
            Reset View
          </button>
        </div>
        <div className="text-xs text-muted-foreground">
          Zoom: {(zoom * 100).toFixed(0)}% | Mouse wheel: zoom | Drag: pan
        </div>
      </div>

      <div 
        className="relative w-full aspect-square border rounded-lg bg-muted/10 p-8 cursor-move"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg 
          ref={svgRef}
          className="w-full h-full" 
          viewBox={`${viewBoxX} ${viewBoxY} ${viewBoxSize} ${viewBoxSize}`}
        >
          {/* Grid */}
          {[0, 25, 50, 75, 100].map((pos) => (
            <g key={pos}>
              <line
                x1={pos}
                y1="0"
                x2={pos}
                y2="100"
                stroke="currentColor"
                strokeOpacity="0.1"
                strokeWidth="0.3"
              />
              <line
                x1="0"
                y1={pos}
                x2="100"
                y2={pos}
                stroke="currentColor"
                strokeOpacity="0.1"
                strokeWidth="0.3"
              />
            </g>
          ))}
          
          {/* Center lines */}
          <line x1="50" y1="0" x2="50" y2="100" stroke="currentColor" strokeOpacity="0.2" strokeWidth="0.5" />
          <line x1="0" y1="50" x2="100" y2="50" stroke="currentColor" strokeOpacity="0.2" strokeWidth="0.5" />
          
          {/* Data points */}
          {data.map((point, index) => {
            const x = ((point.x - xMin) / xRange) * 100;
            const y = 100 - ((point.y - yMin) / yRange) * 100;
            
            return (
              <g key={index}>
                <circle
                  cx={x}
                  cy={y}
                  r={point.isOutlier ? "3" : "2"}
                  fill={point.isOutlier ? "#ef4444" : "#3b82f6"}
                  fillOpacity={point.isOutlier ? "0.8" : "0.6"}
                  stroke={point.isOutlier ? "#dc2626" : "#2563eb"}
                  strokeWidth="0.5"
                >
                  <title>{point.symbol}: ({point.x.toFixed(1)}%, {point.y.toFixed(1)}%)</title>
                </circle>
                {/* Symbol labels for outliers */}
                {point.isOutlier && (
                  <text
                    x={x}
                    y={y - 5}
                    textAnchor="middle"
                    fontSize="1.5"
                    fill="currentColor"
                    className="text-xs font-medium"
                  >
                    {point.symbol}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
        
        {/* Axis labels */}
        <div className="absolute bottom-0 left-0 right-0 text-center text-xs text-muted-foreground">
          {xLabel}
        </div>
        <div className="absolute top-0 left-0 bottom-0 flex items-center">
          <span className="text-xs text-muted-foreground -rotate-90">{yLabel}</span>
        </div>
        
        {/* Legend */}
        <div className="absolute top-2 right-2 text-xs space-y-1 bg-background/80 p-2 rounded border">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
            <span>Normal</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-red-500"></div>
            <span>Outlier</span>
          </div>
        </div>
      </div>
      
      <div className="flex justify-between text-xs text-muted-foreground px-2">
        <span>{xMin.toFixed(1)}%</span>
        <span>{xMax.toFixed(1)}%</span>
      </div>
    </div>
  );
}


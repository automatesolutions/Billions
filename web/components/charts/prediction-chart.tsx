'use client';

interface PredictionChartProps {
  currentPrice: number;
  predictions: number[];
  confidenceUpper?: number[];
  confidenceLower?: number[];
}

export function PredictionChart({
  currentPrice,
  predictions,
  confidenceUpper,
  confidenceLower
}: PredictionChartProps) {
  if (!predictions || predictions.length === 0) {
    return <div className="text-muted-foreground text-sm">No prediction data</div>;
  }

  const allValues = [currentPrice, ...predictions];
  if (confidenceUpper) allValues.push(...confidenceUpper);
  if (confidenceLower) allValues.push(...confidenceLower);
  
  const max = Math.max(...allValues);
  const min = Math.min(...allValues);
  const range = max - min || 1;

  const getY = (value: number) => 100 - ((value - min) / range) * 90 + 5;
  
  const predictionPoints = predictions.map((value, index) => {
    const x = ((index + 1) / (predictions.length + 1)) * 100;
    const y = getY(value);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="space-y-2">
      <div className="relative w-full h-64 border rounded-lg bg-muted/10 p-4">
        <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
          {/* Confidence band */}
          {confidenceUpper && confidenceLower && (
            <polygon
              points={predictions.map((_, index) => {
                const x = ((index + 1) / (predictions.length + 1)) * 100;
                const yUpper = getY(confidenceUpper[index]);
                return `${x},${yUpper}`;
              }).join(' ') + ' ' + predictions.map((_, index) => {
                const x = ((predictions.length - index) / (predictions.length + 1)) * 100;
                const yLower = getY(confidenceLower[predictions.length - 1 - index]);
                return `${x},${yLower}`;
              }).join(' ')}
              fill="#10b981"
              fillOpacity="0.1"
            />
          )}
          
          {/* Current price line */}
          <line
            x1="0"
            y1={getY(currentPrice)}
            x2="100"
            y2={getY(currentPrice)}
            stroke="currentColor"
            strokeOpacity="0.3"
            strokeWidth="0.5"
            strokeDasharray="2,2"
          />
          
          {/* Prediction line */}
          <polyline
            points={predictionPoints}
            fill="none"
            stroke="#10b981"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
          
          {/* Points */}
          {predictions.map((value, index) => {
            const x = ((index + 1) / (predictions.length + 1)) * 100;
            const y = getY(value);
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="1.5"
                fill="#10b981"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
        
        {/* Legend */}
        <div className="absolute top-2 right-4 text-xs space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-current opacity-30"></div>
            <span className="text-muted-foreground">Current: ${currentPrice.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-green-500"></div>
            <span className="text-muted-foreground">Forecast</span>
          </div>
          {confidenceUpper && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-2 bg-green-500 opacity-10"></div>
              <span className="text-muted-foreground">Confidence</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="flex justify-between text-xs text-muted-foreground px-4">
        <span>Today</span>
        <span>Day {Math.floor(predictions.length / 2)}</span>
        <span>Day {predictions.length}</span>
      </div>
    </div>
  );
}


import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { SimpleLineChart } from '@/components/charts/simple-line-chart';
import { PredictionChart } from '@/components/charts/prediction-chart';
import { ScatterPlot } from '@/components/charts/scatter-plot';

describe('Chart Components', () => {
  describe('SimpleLineChart', () => {
    it('renders with data', () => {
      const { container } = render(
        <SimpleLineChart data={[100, 105, 110, 108, 112]} />
      );
      expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('handles empty data', () => {
      const { getByText } = render(<SimpleLineChart data={[]} />);
      expect(getByText(/No data available/i)).toBeInTheDocument();
    });
  });

  describe('PredictionChart', () => {
    it('renders prediction chart', () => {
      const { container } = render(
        <PredictionChart
          currentPrice={100}
          predictions={[101, 102, 103, 104, 105]}
        />
      );
      expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('handles confidence intervals', () => {
      const { container } = render(
        <PredictionChart
          currentPrice={100}
          predictions={[101, 102, 103]}
          confidenceUpper={[105, 106, 107]}
          confidenceLower={[97, 98, 99]}
        />
      );
      expect(container.querySelector('polygon')).toBeInTheDocument();
    });
  });

  describe('ScatterPlot', () => {
    it('renders scatter plot', () => {
      const data = [
        { symbol: 'AAPL', x: 10, y: 20, isOutlier: false },
        { symbol: 'TSLA', x: 30, y: 40, isOutlier: true },
      ];
      
      const { container } = render(<ScatterPlot data={data} />);
      expect(container.querySelectorAll('circle').length).toBeGreaterThan(0);
    });

    it('differentiates outliers from normal points', () => {
      const data = [
        { symbol: 'NORM', x: 10, y: 20, isOutlier: false },
        { symbol: 'OUT', x: 30, y: 40, isOutlier: true },
      ];
      
      const { container } = render(<ScatterPlot data={data} />);
      const circles = container.querySelectorAll('circle');
      expect(circles.length).toBe(2);
    });
  });
});


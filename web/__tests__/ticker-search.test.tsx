import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TickerSearch } from '@/components/ticker-search';

// Mock next/navigation
const mockPush = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

describe('TickerSearch Component', () => {
  it('renders search input and button', () => {
    render(<TickerSearch />);
    
    expect(screen.getByPlaceholderText(/Enter ticker/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Analyze/i })).toBeInTheDocument();
  });

  it('updates input value when typing', async () => {
    const user = userEvent.setup();
    render(<TickerSearch />);
    
    const input = screen.getByPlaceholderText(/Enter ticker/i) as HTMLInputElement;
    await user.type(input, 'TSLA');
    
    expect(input.value).toBe('TSLA');
  });

  it('navigates to analyze page on submit', async () => {
    const user = userEvent.setup();
    render(<TickerSearch />);
    
    const input = screen.getByPlaceholderText(/Enter ticker/i);
    const button = screen.getByRole('button', { name: /Analyze/i });
    
    await user.type(input, 'aapl');
    await user.click(button);
    
    expect(mockPush).toHaveBeenCalledWith('/analyze/AAPL');
  });

  it('converts ticker to uppercase', async () => {
    const user = userEvent.setup();
    render(<TickerSearch />);
    
    const input = screen.getByPlaceholderText(/Enter ticker/i);
    await user.type(input, 'tsla{Enter}');
    
    expect(mockPush).toHaveBeenCalledWith('/analyze/TSLA');
  });

  it('does not navigate with empty ticker', async () => {
    const user = userEvent.setup();
    render(<TickerSearch />);
    
    mockPush.mockClear();
    const button = screen.getByRole('button', { name: /Analyze/i });
    await user.click(button);
    
    expect(mockPush).not.toHaveBeenCalled();
  });
});


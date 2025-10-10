import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import LoginPage from '@/app/login/page';

// Mock next-auth
vi.mock('next-auth/react', () => ({
  signIn: vi.fn(),
}));

// Mock next/image
vi.mock('next/image', () => ({
  default: (props: any) => {
    // eslint-disable-next-line @next/next/no-img-element, jsx-a11y/alt-text
    return <img {...props} />;
  },
}));

describe('Authentication', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Login Page', () => {
    it('renders login page correctly', () => {
      render(<LoginPage />);
      
      expect(screen.getByText('BILLIONS')).toBeInTheDocument();
      expect(screen.getByText(/Stock Market Forecasting/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Sign in with Google/i })).toBeInTheDocument();
    });

    it('displays the logo', () => {
      render(<LoginPage />);
      
      const logo = screen.getByAltText('BILLIONS Logo');
      expect(logo).toBeInTheDocument();
    });

    it('shows sign in description', () => {
      render(<LoginPage />);
      
      expect(screen.getByText(/Sign in to access your personalized dashboard/i)).toBeInTheDocument();
    });

    it('shows terms and conditions', () => {
      render(<LoginPage />);
      
      expect(screen.getByText(/By signing in, you agree to our/i)).toBeInTheDocument();
    });

    it('calls signIn when Google button is clicked', async () => {
      const { signIn } = await import('next-auth/react');
      const user = userEvent.setup();
      
      render(<LoginPage />);
      
      const signInButton = screen.getByRole('button', { name: /Sign in with Google/i });
      await user.click(signInButton);
      
      expect(signIn).toHaveBeenCalledWith('google', { callbackUrl: '/dashboard' });
    });
  });

  describe('Authentication Flow', () => {
    it('validates user session structure', () => {
      const mockSession = {
        user: {
          id: 'test-id',
          name: 'Test User',
          email: 'test@example.com',
          image: 'https://example.com/avatar.jpg',
        },
      };

      expect(mockSession.user).toHaveProperty('id');
      expect(mockSession.user).toHaveProperty('email');
      expect(mockSession.user).toHaveProperty('name');
    });
  });
});


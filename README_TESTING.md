# BILLIONS Testing Guide

## 🧪 Testing Overview

BILLIONS uses a comprehensive testing strategy with multiple testing frameworks to ensure code quality and reliability.

## Testing Stack

### Backend Testing
- **Framework**: pytest 8.4.2
- **Coverage**: pytest-cov
- **Async Support**: pytest-asyncio
- **API Testing**: httpx + pytest-httpx
- **Mocking**: pytest-mock

### Frontend Testing
- **Unit Tests**: Vitest 3.2.4
- **Component Testing**: @testing-library/react
- **E2E Tests**: Playwright 1.56.0
- **API Mocking**: MSW 2.11.5

## Running Tests

### Backend Tests

```bash
# Run all backend tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest api/tests/test_main.py

# Run with coverage report
pytest --cov=api --cov-report=html

# Run only fast tests (exclude slow)
pytest -m "not slow"
```

**Expected Output**:
```
======================== 9 passed in 0.48s ========================
Coverage: 90%
```

### Frontend Tests

```bash
cd web

# Run unit tests
pnpm test

# Run tests with UI
pnpm test:ui

# Run tests with coverage
pnpm test:coverage

# Watch mode (for development)
pnpm test
```

**Expected Output**:
```
✓ __tests__/example.test.tsx (3 tests) 3ms
Test Files  1 passed (1)
Tests  3 passed (3)
```

### E2E Tests

```bash
cd web

# Run all E2E tests
pnpm test:e2e

# Run with UI mode
pnpm test:e2e:ui

# Run specific browser
pnpm exec playwright test --project=chromium

# Debug mode
pnpm exec playwright test --debug
```

## Writing Tests

### Backend Test Example

Create test in `api/tests/`:

```python
# api/tests/test_your_feature.py
import pytest

def test_your_endpoint(client):
    """Test description"""
    # Arrange
    data = {"key": "value"}
    
    # Act
    response = client.post("/api/v1/your-endpoint", json=data)
    
    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_with_database(db_session):
    """Test with database access"""
    from db.models import YourModel
    
    # Create test data
    item = YourModel(name="test")
    db_session.add(item)
    db_session.commit()
    
    # Test your logic
    result = db_session.query(YourModel).first()
    assert result.name == "test"
```

### Frontend Test Example

Create test in `web/__tests__/`:

```typescript
// web/__tests__/your-component.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import YourComponent from '@/components/YourComponent';

describe('YourComponent', () => {
  it('renders correctly', () => {
    render(<YourComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });

  it('handles user interaction', async () => {
    const { user } = render(<YourComponent />);
    const button = screen.getByRole('button');
    
    await user.click(button);
    
    expect(screen.getByText('Clicked')).toBeInTheDocument();
  });
});
```

### E2E Test Example

Create test in `web/e2e/`:

```typescript
// web/e2e/user-flow.spec.ts
import { test, expect } from '@playwright/test';

test.describe('User Flow', () => {
  test('completes full user journey', async ({ page }) => {
    // Navigate to homepage
    await page.goto('/');
    
    // Interact with elements
    await page.getByRole('button', { name: /login/i }).click();
    
    // Fill form
    await page.getByLabel('Email').fill('user@example.com');
    await page.getByLabel('Password').fill('password123');
    await page.getByRole('button', { name: /submit/i }).click();
    
    // Verify outcome
    await expect(page).toHaveURL('/dashboard');
    await expect(page.getByText('Welcome')).toBeVisible();
  });
});
```

## Test Fixtures

### Backend Fixtures (conftest.py)

```python
@pytest.fixture
def client(test_db):
    """FastAPI test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session(test_db):
    """Database session"""
    session = test_db()
    yield session
    session.close()
```

### Frontend Test Setup (vitest.setup.ts)

```typescript
import '@testing-library/jest-dom';
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});
```

## Coverage Reports

### Generate Coverage Reports

```bash
# Backend coverage
pytest --cov=api --cov-report=html
# Open: htmlcov/index.html

# Frontend coverage
cd web && pnpm test:coverage
# Open: coverage/index.html
```

### Coverage Targets

- **Backend**: >80% coverage
- **Frontend**: >70% coverage
- **Critical Paths**: 100% coverage

## CI/CD Integration

Tests run automatically on:
- Every pull request
- Every push to `main` or `develop`
- Manual workflow dispatch

### GitHub Actions Workflows

**Test Workflow** (`.github/workflows/test.yml`):
- Backend tests with coverage
- Frontend unit tests
- E2E tests with Playwright
- Coverage upload to Codecov

**Lint Workflow** (`.github/workflows/lint.yml`):
- Backend linting (flake8, black, isort)
- Frontend linting (ESLint)

## Pre-commit Hooks

Install hooks:
```bash
pre-commit install
```

Run manually:
```bash
pre-commit run --all-files
```

Hooks include:
- black (Python formatting)
- isort (Import sorting)
- flake8 (Python linting)
- ESLint (TypeScript/React linting)
- trailing whitespace removal
- YAML/JSON validation

## Test Organization

```
Billions/
├── api/
│   └── tests/
│       ├── conftest.py          # Shared fixtures
│       ├── test_main.py         # Main API tests
│       ├── test_market.py       # Market endpoints
│       └── test_*.py            # Feature tests
│
└── web/
    ├── __tests__/               # Unit tests
    │   └── *.test.tsx
    └── e2e/                     # E2E tests
        └── *.spec.ts
```

## Testing Best Practices

### 1. Test Naming
- Use descriptive names: `test_user_can_create_prediction()`
- Follow pattern: `test_<what>_<condition>_<expected>`

### 2. Test Structure (AAA Pattern)
```python
def test_example():
    # Arrange - Setup test data
    data = create_test_data()
    
    # Act - Execute the code
    result = function_under_test(data)
    
    # Assert - Verify the outcome
    assert result == expected_value
```

### 3. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 4. Test Data
- Use factories or fixtures
- Keep test data minimal and focused
- Use meaningful data that clarifies the test

### 5. Assertions
- One logical assertion per test
- Use descriptive assertion messages
- Test both success and failure cases

## Debugging Tests

### Backend Debugging

```bash
# Run with pdb debugger
pytest --pdb

# Stop on first failure
pytest -x

# Capture print statements
pytest -s
```

### Frontend Debugging

```bash
# Run tests in watch mode
cd web && pnpm test

# Debug specific test
cd web && pnpm test --run your-test.test.tsx
```

### E2E Debugging

```bash
# Run with UI mode
cd web && pnpm test:e2e:ui

# Run in debug mode
cd web && pnpm exec playwright test --debug

# Generate trace
cd web && pnpm exec playwright test --trace on
```

## Common Issues

### Backend

**Issue**: Tests fail with database errors
**Solution**: Check that test database is properly isolated

```python
# Use test_db fixture
def test_example(test_db, client):
    ...
```

### Frontend

**Issue**: Component tests fail with "not wrapped in act()"
**Solution**: Use async/await with user interactions

```typescript
await user.click(button);  // ✅ Correct
user.click(button);        // ❌ Wrong
```

### E2E

**Issue**: Tests are flaky
**Solution**: Use proper waiting strategies

```typescript
// ✅ Wait for element
await page.waitForSelector('.my-element');

// ✅ Use built-in assertions
await expect(page.locator('.my-element')).toBeVisible();
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Vitest Documentation](https://vitest.dev/)
- [Testing Library](https://testing-library.com/)
- [Playwright Documentation](https://playwright.dev/)
- [PHASE2_SUMMARY.md](./PHASE2_SUMMARY.md) - Detailed Phase 2 completion summary

## Quick Reference

### Run All Tests
```bash
# Backend
pytest

# Frontend
cd web && pnpm test && pnpm test:e2e
```

### Check Coverage
```bash
# Backend
pytest --cov=api

# Frontend
cd web && pnpm test:coverage
```

### Format & Lint
```bash
# Backend
black api/ && isort api/ && flake8 api/

# Frontend
cd web && pnpm lint:fix
```

---

**Happy Testing! 🧪✅**


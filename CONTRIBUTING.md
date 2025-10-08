# Contributing to BILLIONS ML Prediction System

Thank you for your interest in contributing to BILLIONS! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- Clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Error messages and stack traces
- Screenshots if applicable

### Suggesting Enhancements

We love new ideas! For feature requests:

- Use a clear and descriptive title
- Provide detailed description of the proposed feature
- Explain why this feature would be useful
- Include examples or mockups if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/Billions.git
   cd Billions
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow the coding style (PEP 8)
   - Add docstrings to functions
   - Update documentation if needed
   - Add tests for new features

4. **Test your changes**
   ```bash
   python -m pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes

## 📝 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Keep functions focused and small
- Maximum line length: 100 characters

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings

```python
def example_function(param1, param2):
    """
    Brief description of the function.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        
    Returns:
        type: Description of return value
        
    Raises:
        ValueError: When input is invalid
    """
    pass
```

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests when relevant

Examples:
```
Add LSTM model for cryptocurrency prediction
Fix bug in outlier detection algorithm
Update documentation for enhanced features
Refactor database connection handling
```

## 🧪 Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Test edge cases and error conditions

## 🌳 Branch Naming

Use descriptive branch names:

- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Critical fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

Examples:
```
feature/add-crypto-support
bugfix/fix-lstm-nan-values
docs/update-installation-guide
refactor/optimize-feature-engineering
```

## 📦 Adding Dependencies

If your contribution requires new dependencies:

1. Add them to `requirements.txt`
2. Document why they're needed
3. Use specific version numbers
4. Consider the license compatibility

## 🔍 Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Keep the discussion respectful and constructive
4. Be patient - reviews may take time

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **New prediction models** (GRU, Transformer, etc.)
- **Additional technical indicators**
- **Backtesting framework**
- **Performance optimizations**
- **UI/UX improvements**
- **Documentation and examples**
- **Test coverage**
- **Bug fixes**

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behavior:**
- Harassment, trolling, or insulting comments
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 📞 Questions?

Feel free to:
- Open an issue for questions
- Start a discussion in GitHub Discussions
- Reach out to the maintainers

## 🙏 Thank You!

Your contributions make BILLIONS better for everyone. We appreciate your time and effort!

---

**Happy Coding! 🚀**


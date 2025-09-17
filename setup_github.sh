#!/bin/bash
# GitHub Setup Script for Markov Chain of ER Graphs

echo "🚀 Setting up GitHub repository for Markov Chain of ER Graphs..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Add all files
echo "📁 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Markov Chain of ER Graphs - Greedy Reordering

- Complete Monte Carlo simulation framework
- Multiple distance modes (noisy, hamming, edgecount)  
- GPU acceleration with PyTorch
- Debug tools and comprehensive analysis
- Professional package structure with documentation
- 833 lines of clean, well-documented code
- Cross-platform support (CPU, CUDA, MPS)
- Research-ready with step-by-step analysis tools"

# Create GitHub Actions workflow
echo "⚙️ Setting up GitHub Actions..."
mkdir -p .github/workflows

cat > .github/workflows/test.yml << 'EOF'
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Test basic functionality
      run: |
        python src/greedy_reorder_mc.py --n 20 --T 5 --trials 3 --quiet
        python src/compare_modes.py --n 20 --T 5 --trials 3 --quiet
        python src/debug_trial.py --n 20 --T 5 --quiet
    - name: Test package installation
      run: |
        pip install -e .
        python -c "from src import run_mc; print('Package import successful')"
EOF

# Add GitHub Actions to git
git add .github/
git commit -m "Add GitHub Actions CI workflow for testing across Python versions"

# Set up main branch
git branch -M main

echo "✅ Git repository ready!"
echo ""
echo "🔗 Next steps:"
echo "1. Go to https://github.com/new"
echo "2. Create repository named: markov-chain-of-er-graphs"
echo "3. Description: Greedy reconstruction algorithms for sequences of Erdős-Rényi graphs with noisy permutations"
echo "4. Make it PUBLIC"
echo "5. DON'T initialize with README (we have one)"
echo "6. Click 'Create repository'"
echo ""
echo "7. Then run these commands:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/markov-chain-of-er-graphs.git"
echo "   git push -u origin main"
echo ""
echo "8. Add these topics to your repository:"
echo "   - graph-theory"
echo "   - markov-chains" 
echo "   - greedy-algorithms"
echo "   - pytorch"
echo "   - monte-carlo"
echo "   - research"
echo "   - python"
echo ""
echo "🎉 Your project will be live on GitHub!"

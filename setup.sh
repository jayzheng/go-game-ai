#!/bin/bash

# Setup script for Go AI Game
# This script sets up both backend and frontend

set -e  # Exit on error

echo "========================================="
echo "Go AI Game - Setup Script"
echo "========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16 or higher."
    exit 1
fi

echo "✅ Node.js found: $(node --version)"
echo ""

# Backend setup
echo "========================================="
echo "Setting up Backend..."
echo "========================================="

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✅ Backend setup complete!"
echo ""

cd ..

# Frontend setup
echo "========================================="
echo "Setting up Frontend..."
echo "========================================="

cd frontend

# Install dependencies
echo "Installing npm dependencies..."
npm install --silent

echo "✅ Frontend setup complete!"
echo ""

cd ..

# Done
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Open browser to: http://localhost:3000"
echo ""
echo "To train AI model:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   python train.py --iterations 10 --games 50 --simulations 100"
echo ""
echo "For more information, see README.md and TRAINING_GUIDE.md"
echo "========================================="

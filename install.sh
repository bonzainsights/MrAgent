#!/bin/bash
# MRAgent Installer for macOS/Linux
# 
# Installs MRAgent locally in a virtual environment.
# Adds a 'mragent' command to your shell.

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      MRAgent Installer v0.1.0          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# 1. Check for Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
    # Check version
    ver=$($PYTHON_CMD -c"import sys; print(sys.version_info.major)")
    if [ $ver -ne 3 ]; then
        echo -e "${RED}❌ Python 3 is required. Found Python 2.${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Python 3 not found. Please install it first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found Python: $($PYTHON_CMD --version)${NC}"

# 2. Setup Virtual Environment
echo -e "${BLUE}🔧 Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "   Correctly created .venv"
else
    echo "   Found existing .venv"
fi

# 3. Install Dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
source .venv/bin/activate
pip install --upgrade pip > /dev/null
if pip install -r requirements.txt; then
    echo -e "${GREEN}✅ Dependencies installed!${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies.${NC}"
    exit 1
fi

# 4. Create Launcher Script
LAUNCHER_PATH="$(pwd)/mragent_launcher.sh"
echo "#!/bin/bash" > "$LAUNCHER_PATH"
echo "cd \"$(pwd)\"" >> "$LAUNCHER_PATH"
echo "source .venv/bin/activate" >> "$LAUNCHER_PATH"
echo "python main.py \"\$@\"" >> "$LAUNCHER_PATH"
chmod +x "$LAUNCHER_PATH"

# 5. Add to Path (Alias)
echo -e "${BLUE}🔗 Creating 'mragent' alias...${NC}"
SHELL_CONFIG=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
fi

if [ -n "$SHELL_CONFIG" ]; then
    # Remove old alias if exists
    grep -v "alias mragent=" "$SHELL_CONFIG" > "${SHELL_CONFIG}.tmp" && mv "${SHELL_CONFIG}.tmp" "$SHELL_CONFIG"
    
    # Add new alias
    echo "alias mragent='\"$LAUNCHER_PATH\"'" >> "$SHELL_CONFIG"
    echo -e "${GREEN}✅ Added alias to $SHELL_CONFIG${NC}"
    echo -e "   Run 'source $SHELL_CONFIG' or open a new terminal to use it."
else
    echo -e "${RED}⚠️  Could not find shell config (.zshrc/.bashrc).${NC}"
    echo "   You can run manually with: ./mragent_launcher.sh"
fi

echo ""
echo -e "${GREEN}🎉 Installation Complete!${NC}"
echo -e "Type ${BLUE}mragent${NC} to start your assistant."

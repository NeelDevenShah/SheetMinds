#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    local port="$1"
    if command_exists lsof; then
        lsof -i ":$port" >/dev/null 2>&1
    elif command_exists netstat; then
        netstat -an | grep -q ":$port .*LISTEN"
    else
        # Fallback to checking if the port is open
        if command_exists python3; then
            python3 -c "import socket; s = socket.socket(); s.settimeout(1); 
            try: s.connect(('localhost', $port)); s.close(); exit(0); 
            except: exit(1)" 2>/dev/null
            return $?
        else
            echo -e "${YELLOW}Warning: Could not check if port $port is in use${NC}"
            return 1
        fi
    fi
}

# Function to start the API server
start_server() {
    local port=5000
    
    if port_in_use $port; then
        echo -e "${YELLOW}Port $port is already in use. Trying to find an available port...${NC}"
        while port_in_use $port; do
            port=$((port + 1))
        done
        echo -e "${GREEN}Found available port: $port${NC}"
        export FLASK_RUN_PORT=$port
    fi
    
    echo -e "${GREEN}Starting SheetMind API server on port ${FLASK_RUN_PORT:-$port}...${NC}"
    
    # Run the Flask app in the background
    python3 -u api.py > api.log 2>&1 & 
    SERVER_PID=$!
    
    # Wait for the server to start
    echo -n "Waiting for server to start"
    for _ in {1..10}; do
        if port_in_use "${FLASK_RUN_PORT:-$port}"; then
            echo -e "\n${GREEN}Server is running on port ${FLASK_RUN_PORT:-$port} (PID: $SERVER_PID)${NC}"
            echo "API logs are being written to api.log"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo -e "\n${RED}Failed to start server. Check api.log for details.${NC}"
    return 1
}

# Function to stop the server
stop_server() {
    local port=${FLASK_RUN_PORT:-5000}
    local pid
    
    if command_exists lsof; then
        pid=$(lsof -t -i ":$port" 2>/dev/null)
    elif command_exists netstat; then
        pid=$(netstat -tulpn 2>/dev/null | grep ":$port" | grep -oP '\d+(?=/python3?)' | head -1)
    fi
    
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Stopping server (PID: $pid)...${NC}"
        kill -TERM "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
        echo -e "${GREEN}Server stopped.${NC}"
    else
        echo -e "${YELLOW}No running server found on port $port.${NC}"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "api.log" ]; then
        echo -e "${GREEN}Showing logs (press Ctrl+C to stop):${NC}"
        tail -f api.log
    else
        echo -e "${YELLOW}No log file found.${NC}"
    fi
}

# Function to install dependencies
install_deps() {
    echo -e "${GREEN}Installing dependencies...${NC}"
    
    # Check if Python 3 is installed
    if ! command_exists python3; then
        echo -e "${RED}Python 3 is required but not installed. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command_exists pip3; then
        echo -e "${YELLOW}pip3 not found. Installing pip3...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y python3-pip
        elif command_exists yum; then
            sudo yum install -y python3-pip
        elif command_exists brew; then
            brew install python
        else
            echo -e "${RED}Could not determine package manager. Please install pip3 manually.${NC}"
            exit 1
        fi
    fi
    
    # Install dependencies
    pip3 install -r requirements.txt
    
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

# Main menu
main_menu() {
    echo -e "\n${GREEN}SheetMind API Control Panel${NC}"
    echo "1. Start API server"
    echo "2. Stop API server"
    echo "3. Show logs"
    echo "4. Install dependencies"
    echo "5. Exit"
    echo -n "Choose an option (1-5): "
    
    read -r choice
    case $choice in
        1)
            start_server
            ;;
        2)
            stop_server
            ;;
        3)
            show_logs
            ;;
        4)
            install_deps
            ;;
        5)
            stop_server
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
}

# Check if an argument was provided
if [ $# -eq 0 ]; then
    # Interactive mode
    while true; do
        main_menu
    done
else
    # Command-line mode
    case $1 in
        start)
            start_server
            ;;
stop)
            stop_server
            ;;
        logs)
            show_logs
            ;;
        install)
            install_deps
            ;;
        *)
            echo "Usage: $0 [start|test|stop|logs|install]"
            exit 1
            ;;
    esac
fi

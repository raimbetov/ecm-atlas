#!/bin/bash

# Run 4 agents in parallel for pitch deck task
# 2 Claude Code + 2 Codex

TASK_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$TASK_FILE" ]; then
    echo "❌ Usage: ./run_4_agents.sh 01_task_beautiful_pitchdeck.md"
    exit 1
fi

if [ ! -f "$TASK_FILE" ]; then
    echo "❌ Task file not found: $TASK_FILE"
    exit 1
fi

TASK_PATH="$(cd "$(dirname "$TASK_FILE")" && pwd)/$(basename "$TASK_FILE")"

echo "🚀 Launching 4 agents in parallel..."
echo "📄 Task: $TASK_PATH"
echo ""

# Function to run Claude Code agent
run_claude() {
    AGENT_NAME=$1
    WORKSPACE=$2

    echo "🤖 Starting $AGENT_NAME in $WORKSPACE..."

    cd "$WORKSPACE" || exit 1

    # Create symlink to task file
    ln -sf "$TASK_PATH" ./01_task.md

    # Run Claude Code CLI
    claude-code --non-interactive --task ./01_task.md > claude_output.log 2>&1 &

    CLAUDE_PID=$!
    echo "$AGENT_NAME PID: $CLAUDE_PID"
    echo $CLAUDE_PID > claude.pid

    cd "$SCRIPT_DIR" || exit 1
}

# Function to run Codex agent
run_codex() {
    AGENT_NAME=$1
    WORKSPACE=$2

    echo "🤖 Starting $AGENT_NAME in $WORKSPACE..."

    cd "$WORKSPACE" || exit 1

    # Create symlink to task file
    ln -sf "$TASK_PATH" ./01_task.md

    # Run Codex CLI (assuming codex command exists)
    codex --non-interactive --task ./01_task.md > codex_output.log 2>&1 &

    CODEX_PID=$!
    echo "$AGENT_NAME PID: $CODEX_PID"
    echo $CODEX_PID > codex.pid

    cd "$SCRIPT_DIR" || exit 1
}

# Launch all 4 agents
run_claude "Claude Code 01" "$SCRIPT_DIR/claude_code_01"
sleep 2
run_claude "Claude Code 02" "$SCRIPT_DIR/claude_code_02"
sleep 2
run_codex "Codex 01" "$SCRIPT_DIR/codex_01"
sleep 2
run_codex "Codex 02" "$SCRIPT_DIR/codex_02"

echo ""
echo "✅ All 4 agents launched!"
echo ""
echo "📊 Monitor progress:"
echo "  - Claude Code 01: tail -f claude_code_01/claude_output.log"
echo "  - Claude Code 02: tail -f claude_code_02/claude_output.log"
echo "  - Codex 01: tail -f codex_01/codex_output.log"
echo "  - Codex 02: tail -f codex_01/codex_output.log"
echo ""
echo "🔍 Check PIDs:"
echo "  ps aux | grep -E 'claude-code|codex'"
echo ""
echo "⏱️  Agents will complete in ~10-30 minutes depending on complexity"
echo "📁 Results will be in: {agent_name}/90_results_{agent_name}.md"

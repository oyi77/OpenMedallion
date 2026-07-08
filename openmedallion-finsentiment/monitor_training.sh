#!/bin/bash
# Monitor training progress every 5 minutes

while true; do
    echo "=== Training Progress Check at $(date) ==="
    
    # Check if process is still running
    if ps -p 541077 > /dev/null 2>&1; then
        echo "✓ Training process (PID 541077) is running"
        
        # Show last 20 lines of log
        echo ""
        echo "Last 20 lines of training.log:"
        tail -20 training.log
        
        # Extract current step if available
        CURRENT_STEP=$(grep -oP 'step \K\d+(?=/1000)' training.log | tail -1)
        if [ -n "$CURRENT_STEP" ]; then
            PROGRESS=$((CURRENT_STEP * 100 / 1000))
            echo ""
            echo "Progress: Step $CURRENT_STEP/1000 ($PROGRESS%)"
        fi
    else
        echo "✗ Training process completed or stopped"
        echo ""
        echo "Final log output:"
        tail -50 training.log
        break
    fi
    
    echo ""
    echo "Next check in 5 minutes..."
    echo "========================================="
    echo ""
    
    sleep 300  # Wait 5 minutes
done

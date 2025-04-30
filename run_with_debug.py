#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback

def main():
    try:
        # Add current directory to path
        sys.path.append(os.getcwd())
        
        # Import and run the main function
        from pirates_optimization.pirates_hyperparameter_tuning import main
        main()
    except Exception as e:
        # Print error details and save to log file
        error_msg = f"ERROR: {str(e)}\n\n"
        error_msg += traceback.format_exc()
        
        print(error_msg)
        
        with open("error_log.txt", "w") as f:
            f.write(error_msg)
        
        print("Error details saved to error_log.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 
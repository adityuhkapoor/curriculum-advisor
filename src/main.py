# src/main.py

import argparse
import logging
import scraper
import index_creator
import chatbot
from utils import load_environment_variables, configure_logging

def main():
    # Configure logging
    configure_logging()

    # Load environment variables
    if not load_environment_variables():
        logging.error("Failed to load environment variables.")
        return

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Curriculum Chatbot CLI')
    parser.add_argument(
        'command',
        choices=['scrape', 'index', 'chatbot'],
        help='Command to execute'
    )
    args = parser.parse_args()

    if args.command == 'scrape':
        logging.info("Starting the scraper...")
        scraper.main()
    elif args.command == 'index':
        logging.info("Creating/updating the index...")
        index_creator.main()
    elif args.command == 'chatbot':
        logging.info("Starting the chatbot...")
        chatbot.chat()
    else:
        logging.error("Invalid command.")

if __name__ == '__main__':
    main()

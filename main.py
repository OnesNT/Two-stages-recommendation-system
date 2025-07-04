from commands.parser_commands import setup_parser
from commands.train_commands import execute as train_execute
from commands.recommend_commands import execute as recommend_execute
from commands.evaluate_commands import execute as evaluate_execute

def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == 'train':
        train_execute(args)
    elif args.command == 'recommend':
        recommend_execute(args)
    elif args.command == 'evaluate':
        evaluate_execute(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
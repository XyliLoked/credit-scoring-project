from credit_scoring import CreditScoringModel
import argparse

def main():
    parser = argparse.ArgumentParser(description='Запуск ML модели кредитного скоринга')
    parser.add_argument('--source', choices=['online', 'local'], default='online',
                    help='Источник данных (online или local)')
    parser.add_argument('--visualize', action='store_true', 
                    help='Показать визуализации')
    
    args = parser.parse_args()
    
    # Запуск проекта
    model = CreditScoringModel()
    model.load_data(source=args.source)
    model.explore_data()
    model.preprocess_data()
    model.train_models()
    
    if args.visualize:
        model.evaluate_models()
    
    model.generate_report()

if __name__ == "__main__":
    main()
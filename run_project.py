from credit_scoring import CreditScoringModel
import argparse

def main():
    parser = argparse.ArgumentParser(description='Запуск ML модели кредитного скоринга')
    parser.add_argument('--source', choices=['online', 'local'], default='online',
                    help='Источник данных (online или local)')
    parser.add_argument('--visualize', action='store_true', 
                    help='Показать статические визуализации')
    parser.add_argument('--dashboard', action='store_true',
                    help='Запустить интерактивный дашборд')
    parser.add_argument('--quick', action='store_true',
                    help='Быстрый запуск без визуализаций')
    
    args = parser.parse_args()
    
    print("🎯 ЗАПУСК ПРОЕКТА КРЕДИТНОГО СКОРИНГА")
    print("=" * 50)
    
    model = CreditScoringModel()
    model.load_data(source=args.source)
    
    if not args.quick:
        model.explore_data()
    
    model.preprocess_data()
    model.train_models()
    
    if args.visualize and not args.quick:
        model.evaluate_models()
    
    if args.dashboard:
        model.create_dashboard()
    else:
        model.generate_report()

if __name__ == "__main__":
    main()
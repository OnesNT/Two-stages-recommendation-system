from utils import load_results, plot_all_metrics, plot_metric_distribution
class RecommenderCommands:
    def visualize_results(self, json_file, output_dir, plot_type='all'):
        """
        Visualize evaluation metrics from a JSON file.

        Args:
            json_file (str or Path): Path to the JSON results file.
            output_dir (str or Path): Directory to save plots.
            plot_type (str): 'all', 'line', or 'distribution'
        """
        results = load_results(json_file)

        if plot_type == 'line' or plot_type == 'all':
            plot_all_metrics(results, output_dir)
        if 'user_metrics' in results and (plot_type == 'distribution' or plot_type == 'all'):
            plot_metric_distribution(results, output_dir)

            plot_metric_distribution(results, output_dir)

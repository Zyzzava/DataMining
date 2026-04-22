from evaluation.evaluator import eval
import os

def execution_pipeline(algo, df, unique_texts, tfidf_matrix):
    algo_name = algo.algo_name
    col_name = getattr(algo, "cluster_col", None)

    print(f"Checking for: '{col_name}' in columns: {df.columns.tolist()[:10]}...")
    if col_name in df.columns:
        print(f"\n[SKIP] {algo_name} already exists in column '{col_name}'.")
    else:
        print(f"\n{'='*50}")
        print(f"Executing Pipeline for: {algo_name}")
        print(f"{'='*50}")

        df, target_col = algo.run_pipeline(df, unique_texts, tfidf_matrix)
        algo.create_report()

        if target_col and target_col in df.columns:
            from preprocessing.preprocessor import FULLY_PROCESSED_PARQUET
            print(f"[INFO] Saving updated labels to {FULLY_PROCESSED_PARQUET}...")
            df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)

def evaluation_pipeline(algo, df, unique_texts, tfidf_matrix):
    algo_name = algo.algo_name
    col_name = getattr(algo, "cluster_col", None)

    # Define the output paths first
    algo_report_out = algo.report_dir
    os.makedirs(algo_report_out, exist_ok=True)
    eval_report_path = os.path.join(algo_report_out, f"evaluation_metrics_{col_name}.txt")

    # Correct Check: Does the text report already exist on disk?
    if os.path.exists(eval_report_path):
        print(f"\n[SKIP] Evaluation for {algo_name} already exists at '{eval_report_path}'.")
    else:
        # Safety Check: We can't evaluate if the clustering column isn't there
        if col_name not in df.columns:
            print(f"\n[ERROR] Cannot evaluate. Column '{col_name}' is missing from the DataFrame.")
        else:
            # Running evaluation and saving the report
            print(f"\n[INFO] Evaluating {algo_name} results and saving report to '{eval_report_path}'...") 
            eval(df=df,
                cluster_col=col_name,
                unique_texts=unique_texts,
                tfidf_matrix=tfidf_matrix,
                sample_frac=0.1,
                output_dir=algo_report_out)
    return algo_report_out
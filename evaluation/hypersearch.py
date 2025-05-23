import argparse
import time
import json
import numpy as np

from evaluation import evaluateSummaries, testSummaries

# In kết quả của từng tổ hợp tham số coef, expand, fill_mode, cùng với F-measure (max, avg, top-5).
def print_result(coef, expand, fill_mode, score, idx=None):
    expansion = f"{expand:.2f}" if expand < 1 else f"{int(expand)}"
    param = f"coef={coef:.2f}, expand={expansion}"
    if fill_mode is not None:
        param += f", fill_mode={fill_mode}"
        
    result = f"avg-f={score[1]:.4f}, top-5={score[2]:.4f}"
    if score[0] is not None:
        result += f", max-f={score[0]:.4f}"
    
    if idx is None:
        print("========================================")
        print(f"Parameter: {param}")
        print(f"Result: {result}")
        print("========================================")
    else:
        display = f"{idx+1}. {param}; {result}"
        print(display)
    

def search():
    parser = argparse.ArgumentParser(description='Evaluate machine learning algorithm summaries.')
    
    parser.add_argument('--original', action='store_true',
                        help='Evaluate the original dataset'
                        )
    # parser.add_argument('--original-folder', type=str,
    #                     help='Path to the folder containing the original dataset'
    #                     )
    
    parser.add_argument('--groundtruth-folder', type=str, required=True,
                        help='Path to the folder containing groundtruth .mat files')
    parser.add_argument('--summary-folder', type=str, required=True,
                        help='Path to the folder containing output .npy files')
    parser.add_argument('--result-folder', type=str, default='result',
                        help='Path to the folder containing the result of evaluation')
    
    parser.add_argument('--mode', type=str, default='frame',
                        choices=['frame', 'fragment', 'shot'],
                        help='Evaluation mode: "frame", "fragment" or "shot"')
    
    # Hệ số tối thiểu, tối đa và bước nhảy cho việc mở rộng keyframe
    parser.add_argument('--min-coef', type=float, default=1.0)
    parser.add_argument('--max-coef', type=float, default=5.0)
    parser.add_argument('--iter-coef', type=float, default=0.5)
    
    # Tối thiểu, tối đa và bước nhảy số lượng frame cần mở rộng quanh keyframe
    parser.add_argument('--min-expand', type=float,
                        help='Int: Minimum number of frames to expand'
                        + '0 < Float < 1: Minimum percentage of selection')
    parser.add_argument('--max-expand', type=float,
                        help='Int > 1: Maximum number of frames to expand'
                        + '0 < Float < 1: Maximum percentage of selection')
    parser.add_argument('--iter-expand', type=float,
                        help='Iteration between min-expand and max-expand')
    
    args = parser.parse_args()
    
    fill_modes = ['linear', 'nearest', 'nearest-up']
    search_results = []
    
    # Duyệt qua các giá trị của coef để thử nghiệm
    for coef in np.arange(args.min_coef,
                          args.max_coef + (args.iter_coef / 2),
                          args.iter_coef):
        # Duyệt qua các giá trị của expand để thử nghiệm
        for expand in np.arange(args.min_expand,
                                args.max_expand + (args.iter_expand / 2),
                                args.iter_expand):
            # Nếu chế độ đánh giá là 'shot', thử tất cả các giá trị fill_mode
            if args.mode == 'shot':
                for fill_mode in fill_modes:
                    score = testSummaries(groundtruth_folder=args.groundtruth_folder,
                                          summary_folder=args.summary_folder,
                                          result_folder=None,
                                          coef=coef,
                                          mode=args.mode,
                                          fill_mode=fill_mode,
                                          expand=expand
                                          )
                    
                    # Lưu kết quả tạm thời cho tổ hợp tham số này
                    search_result = {
                        'coef': float(coef),
                        'expand': float(expand),
                        'fill-mode': fill_mode,
                        'max-f': float(score[0]),
                        'avg-f': float(score[1]),
                        'top-5': float(score[2]),
                    }
                    search_results.append(search_result)
                    
                    print_result(coef, expand, fill_mode, score)
            else:
                if args.original:
                    score = evaluateSummaries(groundtruth_folder=args.groundtruth_folder,
                                              summary_folder=args.summary_folder,
                                              result_folder=None,
                                              coef=coef,
                                              mode=args.mode,
                                              expand=expand
                                              )
                    
                    print_result(coef, expand, None, score)
                elif args.mode == 'frame':
                    score = testSummaries(groundtruth_folder=args.groundtruth_folder,
                                          summary_folder=args.summary_folder,
                                          result_folder=None,
                                          coef=coef,
                                          mode=args.mode,
                                          fill_mode=None,
                                          expand=expand
                                          )
                    
                    print_result(coef, expand, None, score)
            
                search_result = {
                    'coef': float(coef),
                    'expand': float(expand),
                    'max-f': float(score[0]) if score[0] is not None else None,
                    'avg-f': float(score[1]),
                    'top-5': float(score[2]),
                }
                search_results.append(search_result)
            
    # Sort by score
    cmp_field = 'avg-f' if args.original else 'max-f'
    sorted_results = sorted(search_results, key=lambda val: val[cmp_field],
                            reverse=True)
    
        # Sort by avg-f to find the best average F-score
    best_result = max(search_results, key=lambda val: val['avg-f'])

    # In ra top 10 kết quả (giữ nguyên nếu cần)
    for i, sorted_result in enumerate(sorted(search_results, key=lambda val: val['avg-f'], reverse=True)[:10]):
        score = [sorted_result.get('max-f'), sorted_result['avg-f'], sorted_result['top-5']]
        print_result(coef=sorted_result['coef'],
                     expand=sorted_result['expand'],
                     fill_mode=sorted_result.get('fill-mode', None),
                     score=score,
                     idx=i)

    # In ra kết quả tốt nhất theo avg-f
    print("\n>>> BEST RESULT BASED ON AVG-F <<<")
    best_score = [best_result.get('max-f'), best_result['avg-f'], best_result['top-5']]
    print_result(coef=best_result['coef'],
                 expand=best_result['expand'],
                 fill_mode=best_result.get('fill-mode', None),
                 score=best_score)

    # # Lưu file json
    # result_type = 'original' if args.original else 'test'
    # result_name = f'{result_type}_{args.mode}'
    # result_path = args.result_folder + f'/{result_name}_results.json'
    
    # # with open(result_path, 'w', encoding='utf-8') as result_file:
    # #     json.dump(search_results, result_file)

if __name__ == '__main__':
    start_time = time.time()
    search()
    end_time = time.time()
    
    print(f'Time taken: {end_time - start_time:.2f} seconds')

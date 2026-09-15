from pathlib import Path
from time import perf_counter
from ai.router.orchestrator import SatQueryOrchestrator
ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [ROOT / 'data' / 'LEVIR-CD' / 'test' / 'A' / 'test_1.png', ROOT / 'frontend' / 'uploads']

def find_test_image():
    first = CANDIDATES[0]
    if first.exists():
        return first
    uploads = CANDIDATES[1]
    if uploads.exists():
        images = []
        for extension in ['*.png', '*.jpg', '*.jpeg']:
            images.extend(uploads.glob(extension))
        if images:
            images.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return images[0]
    raise FileNotFoundError('No VQA test image found.')

def run_query(orchestrator, image, question):
    start = perf_counter()
    result = orchestrator.execute(question, image_path=str(image))
    elapsed = perf_counter() - start
    execution = result.get('execution', {})
    print()
    print('=' * 70)
    print('QUESTION:', question)
    print('ANSWER:', execution.get('answer'))
    print('SUCCESS:', execution.get('success'))
    print('DEVICE:', execution.get('device'))
    print('CACHE HIT:', execution.get('cache_hit'))
    print('MODEL TIMING:', execution.get('timing_ms'))
    print('TOTAL ORCHESTRATOR TIME:', round(elapsed, 3), 'seconds')
    if not execution.get('success'):
        raise AssertionError(f'VQA inference failed: {execution}')
    if 'total' not in execution.get('timing_ms', {}):
        raise AssertionError('Missing VQA timing telemetry.')
    return elapsed

def main():
    image = find_test_image()
    print('IMAGE:', image)
    orchestrator = SatQueryOrchestrator()
    print()
    print('PRELOADING BLIP...')
    preload_start = perf_counter()
    preload = orchestrator.preload_vqa(warmup=False)
    preload_time = perf_counter() - preload_start
    if not preload.get('success'):
        raise AssertionError(f'VQA preload failed: {preload}')
    print('PRELOAD RESULT:', preload)
    print('PRELOAD TIME:', round(preload_time, 3), 'seconds')
    first = run_query(orchestrator, image, 'Are there roads in this image?')
    second = run_query(orchestrator, image, 'What is visible in this image?')
    repeated = run_query(orchestrator, image, 'Are there roads in this image?')
    print()
    print('=' * 70)
    cached = orchestrator.execute('Are there roads in this image?', image_path=str(image))['execution']
    if not cached.get('cache_hit'):
        raise AssertionError('Repeated VQA query did not use the answer cache.')
    print('PERFORMANCE SUMMARY')
    print('=' * 70)
    print('Model preload:', round(preload_time, 3), 's')
    print('First inference:', round(first, 3), 's')
    print('Second inference:', round(second, 3), 's')
    print('Repeated cached query:', round(repeated, 3), 's')
if __name__ == '__main__':
    main()

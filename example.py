from src.ocra.orientation import OrientationPredictor


def main():
    predictor = OrientationPredictor(
        device="auto",
        threshold=0.95,
        verbose=1
    )
    
    print("=== Single Image Prediction ===")
    single_result = predictor.predict_single(r"examples\hrk_463.png")
    if single_result:
        print(f"File: {single_result['filename']}")
        print(f"Orientation: {single_result['prediction']}")
        print(f"Confidence: {single_result['confidence']:.4f}")
        print(f"High confidence: {single_result['high_confidence']}")
    
    print("\n=== Folder Prediction ===")
    folder_path = "examples"
    results = predictor.predict(folder_path, batch_size=32)
    
    print(f"Total images: {results['summary']['total']}")
    print(f"Processed: {results['summary']['processed']}")
    print(f"Vertical: {results['summary']['vertical']}")
    print(f"Horizontal: {results['summary']['horizontal']}")
    print(f"Low confidence: {results['summary']['low_confidence']}")
    print(f"Average confidence: {results['summary']['avg_confidence']:.4f}")
    
    print("\n=== High Confidence Only ===")
    high_conf = predictor.filter_by_confidence(results, min_confidence=0.98)
    print(f"High confidence (>0.98): {high_conf['summary']['filtered_count']}")
    print(f"Vertical (high conf): {high_conf['summary']['filtered_vertical']}")
    
    vertical_high_conf = [
        r for r in high_conf['results'] 
        if r['pred_class'] == 1 and r['high_confidence']
    ]
    
    print(f"\nVertical images with high confidence: {len(vertical_high_conf)}")
    
    for i, result in enumerate(vertical_high_conf[:5]):
        print(f"{i+1}. {result['filename']} - {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
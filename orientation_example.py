from src.orientation import OrientationPredictor

IMAGE_PATH = r"examples\hrk_463.png"

def main():
    model = OrientationPredictor()

    result = model.predict_path(IMAGE_PATH)
    label = "VERT" if result["pred"] == 1 else "HORZ"

    print("Path:", result["path"])
    print("Pred:", label, f"(class={result['pred']})")
    print(f"Prob VERT: {result['prob_vert']:.4f}")
    print(f"Prob HORZ: {result['prob_horz']:.4f}")
    print(f"Aspect (w/h): {result['aspect']:.4f}")

if __name__ == "__main__":
    main()

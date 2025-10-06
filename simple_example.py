from src.ocra.orientation import OrientationPredictor
from src.ocra.ishandwritten import HandwrittenPredictor

def main():
    orient_pred = OrientationPredictor(device="cpu", verbose=0)
    text_pred = HandwrittenPredictor(verbose=False)
    
    image = "examples/hrk_463.png"
    
    orient_res = orient_pred.predict_single(image)
    orientation = "вертикальная" if orient_res['pred_class'] == 1 else "горизонтальная"
    
    text_res = text_pred.predict_single(image)
    text_type = text_res['prediction']
    
    print(f"   Ориентация: {orientation}")
    print(f"   Тип текста: {text_type}")
    
    print(f"\nАнализ examples/")
    
    import os
    files = [f for f in os.listdir("examples") if f.endswith('.png')][:5]
    
    for file in files:
        path = f"examples/{file}"
        
        o_res = orient_pred.predict_single(path)
        o_type = "↕️" if o_res['pred_class'] == 1 else "↔️"
        
        t_res = text_pred.predict_single(path)
        t_type = "✍️" if t_res['prediction'] == 'handwritten' else "🖨️"
        
        print(f"   {file:18} {o_type} {t_type}")

if __name__ == "__main__":
    main()
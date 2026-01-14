import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    corr, total = 0, 0

    modality_metrics = {}
    category_metrics = {}
    subcat_metrics = {}

    for s in data:
        pred = s.get("model_prediction", "").strip().upper()
        choices = s["choices"]
        answer = s["answer"]

        # ❗ 把 GT 文本 → 字母
        try:
            gt_idx = choices.index(answer)
            gt = "ABCD"[gt_idx]
        except:
            # 如果 answer 不在 choices 里（极少）
            continue

        modality = s["modality"]
        category = s["category"]
        subcat = s.get("sub-category", None)

        modality_metrics.setdefault(modality, [0, 0])
        category_metrics.setdefault(category, [0, 0])
        if subcat:
            subcat_metrics.setdefault(subcat, [0, 0])

        ok = (pred == gt)

        if ok:
            corr += 1
            modality_metrics[modality][0] += 1
            category_metrics[category][0] += 1
            if subcat:
                subcat_metrics[subcat][0] += 1

        modality_metrics[modality][1] += 1
        category_metrics[category][1] += 1
        if subcat:
            subcat_metrics[subcat][1] += 1

        total += 1

    print("*" * 30)
    print("Modality-wise Accuracy:")
    for k in modality_metrics:
        c, t = modality_metrics[k]
        print(f"{k} : {100*c/t:.2f}% over {t}")

    print("*" * 30)
    print("Category-wise Accuracy:")
    for k in category_metrics:
        c, t = category_metrics[k]
        print(f"{k} : {100*c/t:.2f}% over {t}")

    print("*" * 30)
    print("Sub-category-wise Accuracy:")
    for k in subcat_metrics:
        c, t = subcat_metrics[k]
        print(f"{k} : {100*c/t:.2f}% over {t}")

    print("*" * 30)
    print(f"Total Accuracy: {100*corr/total:.2f}% over {total}")

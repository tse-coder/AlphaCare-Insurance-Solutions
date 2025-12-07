"""Generate a synthetic insurance claims dataset for development and testing.

Usage:
    python data/generate_synthetic_data.py --n 10000 --out data/sample_claims.csv
"""
import argparse
import numpy as np
import pandas as pd
import os


def generate(n=10000, seed=42):
    np.random.seed(seed)
    provinces = [
        "Gauteng",
        "Western Cape",
        "KwaZulu-Natal",
        "Eastern Cape",
        "Free State",
        "Limpopo",
        "Mpumalanga",
    ]
    zipcodes = [f"{i:04d}" for i in range(1000, 1100)]  # 100 zipcodes

    policy_id = np.arange(1, n + 1)
    province = np.random.choice(provinces, size=n, p=None)
    zipcode = np.random.choice(zipcodes, size=n)
    gender = np.random.choice(["M", "F"], size=n, p=[0.55, 0.45])
    owner_age = np.random.randint(18, 80, size=n)
    car_age = np.random.randint(0, 20, size=n)
    vehicle_value = np.round(np.random.lognormal(mean=10, sigma=0.6, size=n))

    # baseline risk score
    risk_score = (
        0.01 * (80 - owner_age)
        + 0.02 * car_age
        + 0.00001 * (200000 - vehicle_value)
        + np.random.normal(scale=0.05, size=n)
    )
    # scale to expected claim amount
    total_claims = np.maximum(0.0, np.round(risk_score * vehicle_value * np.random.uniform(0.1, 0.5, n)))

    # premium roughly proportional to vehicle value and risk
    premium = np.round(0.05 * vehicle_value * (1 + risk_score * 2) + np.random.normal(0, 50, n))
    premium = np.maximum(premium, 100.0)

    df = pd.DataFrame(
        {
            "policy_id": policy_id,
            "province": province,
            "zipcode": zipcode,
            "gender": gender,
            "owner_age": owner_age,
            "car_age": car_age,
            "vehicle_value": vehicle_value,
            "total_claims": total_claims,
            "premium": premium,
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--out", type=str, default="data/sample_claims.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(n=args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

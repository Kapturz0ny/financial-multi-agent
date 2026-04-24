import numpy as np

results = {
    "sequential": {
        "AAPL": {
            "stuktura": np.array([70]),
            "bogactwo": np.array([100]),
            "prof": np.array([70]),
            "uzytecznosc": np.array([100]),
            "sentyment": np.array([61]),
            "total": np.array([91.5]),
            "czas": np.array([106.1]),
        },
        "NVDA": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "GOOGL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        }
    },
    "group-chat": {
        "AAPL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "NVDA": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "GOOGL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        }
    },
    "group-chat-cs-rag": {
        "AAPL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "NVDA": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "GOOGL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        }
    },
    "parallel": {
        "AAPL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "NVDA": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "GOOGL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        }
    },
    "concurrent": {
        "AAPL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "NVDA": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        },
        "GOOGL": {
            "stuktura": np.array([]),
            "bogactwo": np.array([]),
            "prof": np.array([]),
            "uzytecznosc": np.array([]),
            "sentyment": np.array([]),
            "total": np.array([]),
            "czas": np.array([]),
        }
    }
}


def generate_raport(results):
    print(f"{'TRYB':<20} | {'SPÓŁKA':<8} | {'METRYKA':<12} | {'ŚREDNIA':<10} | {'STD':<10}")
    print("-" * 70)

    for mode, stocks in results.items():
        for stock, metrics in stocks.items():
            for metric, values in metrics.items():
                if values.size > 0:
                    mean = np.mean(values)
                    std = np.std(values)

                    mean_str = f"{mean:.2f}"
                    std_str = f"{std:.2f}"
                else:
                    mean_str = "null"
                    std_str = "null"

                print(f"{mode:<20} | {stock:<8} | {metric:<12} | {mean_str:<10} | {std_str:<10}")
            print("-" * 70)

if __name__ == "__main__":
    generate_raport(results)

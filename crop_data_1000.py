# 1000 Crops Generator
import numpy as np
import pandas as pd

CORE_ARCHETYPES = [
    ('rice', 80, 48, 40, 24.0, 82.0, 6.4, 236.0),
    ('wheat', 85, 40, 35, 18.5, 55.0, 6.8, 75.0),
    ('maize', 78, 48, 20, 22.4, 65.0, 6.2, 85.0),
    ('barley', 65, 35, 30, 17.5, 52.0, 7.2, 55.0),
    ('oats', 70, 38, 35, 16.5, 60.0, 6.5, 70.0),
    ('sorghum', 75, 35, 30, 26.5, 50.0, 6.8, 60.0),
    ('pearl_millet', 60, 30, 25, 29.0, 42.0, 7.2, 45.0),
    ('finger_millet', 55, 32, 28, 24.0, 62.0, 6.5, 80.0),
    ('foxtail_millet', 50, 28, 25, 25.0, 52.0, 6.7, 55.0),
    ('buckwheat', 45, 35, 30, 18.0, 68.0, 5.8, 90.0),
    ('quinoa', 50, 40, 35, 17.0, 48.0, 6.8, 50.0),
    ('rye', 60, 35, 30, 15.0, 58.0, 6.2, 60.0),
    ('chickpea', 40, 68, 80, 18.9, 16.9, 7.3, 80.0),
    ('kidneybeans', 21, 68, 20, 20.1, 21.6, 5.7, 106.0),
    ('pigeonpeas', 21, 68, 20, 27.7, 48.1, 5.7, 149.0),
    ('mothbeans', 21, 48, 20, 28.2, 53.2, 6.8, 51.0),
    ('mungbean', 21, 47, 20, 28.5, 85.5, 6.7, 48.0),
    ('blackgram', 40, 68, 19, 29.9, 65.0, 7.1, 68.0),
    ('lentil', 19, 68, 19, 24.5, 64.8, 6.9, 46.0),
    ('soybean', 38, 62, 42, 25.0, 65.0, 6.5, 92.0),
    ('cowpea', 25, 45, 30, 27.5, 60.0, 6.4, 75.0),
    ('faba_bean', 30, 55, 40, 18.0, 62.0, 7.0, 85.0),
    ('green_pea', 35, 52, 38, 16.5, 65.0, 6.8, 62.0),
    ('litchi', 62, 32, 50, 26.8, 81.5, 6.3, 188.0),
    ('mango', 20, 27, 30, 31.2, 50.2, 5.8, 94.7),
    ('watermelon', 99, 17, 50, 25.6, 85.2, 6.5, 50.8),
    ('muskmelon', 100, 18, 50, 28.7, 92.3, 6.4, 24.7),
    ('apple', 21, 134, 199, 22.6, 92.3, 5.9, 112.7),
    ('banana', 100, 73, 50, 27.4, 80.4, 5.9, 104.6),
    ('pomegranate', 19, 19, 40, 21.8, 90.1, 6.4, 107.5),
    ('grapes', 23, 133, 200, 23.8, 81.9, 6.0, 69.6),
    ('orange', 20, 17, 10, 22.8, 92.2, 7.0, 110.5),
    ('papaya', 50, 59, 50, 33.7, 92.4, 6.7, 142.6),
    ('guava', 45, 30, 40, 25.5, 70.0, 6.5, 115.0),
    ('strawberry', 40, 50, 70, 16.0, 75.0, 6.0, 82.0),
    ('pineapple', 52, 26, 68, 27.2, 79.5, 5.1, 165.0),
    ('peach', 35, 45, 60, 20.0, 65.0, 6.2, 95.0),
    ('plum', 32, 42, 58, 19.5, 68.0, 6.3, 90.0),
    ('cherry', 38, 48, 62, 17.5, 70.0, 6.4, 88.0),
    ('kiwi', 42, 38, 55, 18.5, 78.0, 6.1, 130.0),
    ('dragonfruit', 45, 35, 50, 28.0, 72.0, 6.6, 85.0),
    ('avocado', 55, 30, 65, 23.0, 65.0, 6.3, 110.0),
    ('fig', 30, 25, 40, 24.5, 55.0, 7.2, 65.0),
    ('jackfruit', 60, 35, 55, 28.5, 78.0, 6.2, 175.0),
    ('custard_apple', 35, 28, 42, 26.0, 60.0, 6.8, 70.0),
    ('potato', 90, 50, 85, 19.0, 68.0, 5.8, 65.0),
    ('tomato', 80, 60, 65, 24.0, 70.0, 6.5, 85.0),
    ('onion', 75, 45, 60, 20.5, 62.0, 6.7, 60.0),
    ('garlic', 60, 40, 50, 17.0, 60.0, 6.5, 50.0),
    ('ginger', 55, 45, 70, 25.0, 80.0, 6.2, 180.0),
    ('cauliflower', 85, 55, 55, 18.0, 72.0, 6.5, 65.0),
    ('cabbage', 90, 50, 60, 17.5, 75.0, 6.6, 68.0),
    ('broccoli', 88, 52, 58, 17.0, 74.0, 6.5, 66.0),
    ('spinach', 80, 40, 50, 16.0, 70.0, 6.8, 50.0),
    ('carrot', 60, 55, 80, 17.0, 65.0, 6.4, 55.0),
    ('radish', 50, 40, 45, 18.5, 68.0, 6.6, 52.0),
    ('brinjal', 72, 48, 52, 25.0, 68.0, 6.4, 82.0),
    ('okra', 65, 45, 45, 27.5, 70.0, 6.7, 95.0),
    ('chilli', 70, 50, 50, 25.0, 65.0, 6.5, 85.0),
    ('bell_pepper', 75, 55, 55, 22.0, 70.0, 6.4, 78.0),
    ('cucumber', 70, 40, 55, 25.5, 75.0, 6.5, 70.0),
    ('pumpkin', 65, 38, 50, 26.0, 72.0, 6.6, 75.0),
    ('bitter_gourd', 58, 35, 45, 27.0, 74.0, 6.5, 80.0),
    ('sweet_potato', 45, 38, 75, 24.0, 70.0, 6.0, 95.0),
    ('cardamom', 45, 30, 55, 19.5, 88.0, 5.5, 240.0),
    ('black_pepper', 50, 30, 60, 26.0, 85.0, 5.8, 230.0),
    ('turmeric', 60, 45, 65, 26.5, 82.0, 6.3, 175.0),
    ('coriander', 45, 35, 30, 19.0, 58.0, 6.8, 55.0),
    ('cumin', 35, 25, 25, 22.0, 45.0, 7.3, 35.0),
    ('fennel', 40, 30, 30, 20.0, 52.0, 7.0, 42.0),
    ('fenugreek', 30, 35, 28, 18.0, 55.0, 7.1, 48.0),
    ('mustard', 70, 35, 30, 18.0, 50.0, 7.0, 45.0),
    ('clove', 50, 28, 55, 27.0, 82.0, 5.8, 210.0),
    ('cinnamon', 48, 25, 50, 26.5, 84.0, 5.6, 220.0),
    ('cotton', 118, 46, 20, 24.0, 79.8, 6.9, 80.4),
    ('jute', 78, 47, 40, 25.0, 79.6, 6.7, 174.8),
    ('coffee', 101, 29, 30, 25.5, 58.9, 6.8, 158.1),
    ('coconut', 22, 17, 31, 27.4, 94.8, 5.9, 175.7),
    ('tea', 70, 25, 40, 21.0, 85.0, 5.2, 210.0),
    ('sugarcane', 130, 55, 60, 28.0, 78.0, 6.8, 170.0),
    ('tobacco', 65, 45, 75, 24.0, 68.0, 6.2, 90.0),
    ('rubber', 60, 30, 45, 27.5, 85.0, 5.4, 250.0),
    ('cocoa', 55, 32, 50, 26.0, 84.0, 6.0, 195.0),
    ('cashew', 40, 25, 35, 27.0, 65.0, 6.2, 120.0),
    ('almond', 50, 40, 60, 21.0, 52.0, 7.1, 60.0),
    ('walnut', 55, 42, 65, 18.0, 58.0, 6.6, 85.0),
    ('sunflower', 65, 50, 45, 24.5, 58.0, 6.8, 70.0),
    ('sesame', 45, 35, 30, 27.0, 55.0, 6.8, 55.0),
    ('groundnut', 35, 55, 40, 27.0, 60.0, 6.4, 75.0),
]

VARIETY_MODS = [
    ('early_season', 0.95, 0.96, 0.95, -0.5, -2.0, 0.0, -5.0),
    ('late_season', 1.05, 1.04, 1.05, 0.6, 2.0, 0.0, 8.0),
    ('hybrid', 1.08, 1.06, 1.05, 0.3, 1.0, 0.0, 2.0),
    ('dwarf', 0.92, 0.95, 0.94, -0.2, 0.0, 0.1, -4.0),
    ('high_yield', 1.12, 1.08, 1.08, 0.2, 1.5, 0.0, 5.0),
    ('organic', 0.88, 0.92, 0.92, -0.4, -1.0, -0.1, -2.0),
    ('drought_hardy', 0.90, 0.90, 0.95, 1.2, -6.0, 0.2, -25.0),
    ('moist_tolerant', 1.04, 1.02, 1.02, -0.5, 5.0, -0.1, 25.0),
    ('hill_variety', 0.88, 0.95, 0.96, -2.5, 4.0, -0.2, 15.0),
    ('plains_variety', 1.06, 1.02, 1.00, 1.5, -3.0, 0.1, -10.0),
    ('sweet_variety', 1.02, 1.05, 1.06, 0.4, 1.0, 0.0, 2.0),
    ('grade_a', 1.05, 1.03, 1.04, 0.0, 0.0, 0.0, 0.0),
    ('export_quality', 1.03, 1.06, 1.08, 0.2, 1.2, 0.0, 4.0),
]

crops_dict = {}
for name, n, p, k, t, h, ph, r in CORE_ARCHETYPES:
    crops_dict[name] = (n, p, k, t, h, ph, r)

mod_i = 0
while len(crops_dict) < 1000:
    for name, n, p, k, t, h, ph, r in CORE_ARCHETYPES:
        if len(crops_dict) >= 1000:
            break
        mod_name, n_m, p_m, k_m, t_m, h_m, ph_m, r_m = VARIETY_MODS[mod_i % len(VARIETY_MODS)]
        cycle = mod_i // len(VARIETY_MODS)
        v_name = f'{name}_{mod_name}' if cycle == 0 else f'{name}_{mod_name}_{cycle+1}'
        v_n = max(5.0, min(140.0, n * n_m + np.random.normal(0, 1.0)))
        v_p = max(5.0, min(145.0, p * p_m + np.random.normal(0, 1.0)))
        v_k = max(5.0, min(205.0, k * k_m + np.random.normal(0, 1.0)))
        v_t = max(9.0, min(43.0, t + t_m + np.random.normal(0, 0.2)))
        v_h = max(15.0, min(99.0, h + h_m + np.random.normal(0, 0.5)))
        v_ph = max(3.8, min(9.5, ph + ph_m + np.random.normal(0, 0.05)))
        v_r = max(20.0, min(295.0, r + r_m + np.random.normal(0, 1.5)))
        crops_dict[v_name] = (v_n, v_p, v_k, v_t, v_h, v_ph, v_r)
        mod_i += 1

rows = []
np.random.seed(42)
samples_per_crop = 25
for cname, (mu_n, mu_p, mu_k, mu_t, mu_h, mu_ph, mu_r) in crops_dict.items():
    sn, sp, sk = max(1.5, mu_n * 0.05), max(1.5, mu_p * 0.05), max(1.5, mu_k * 0.05)
    st, sh, sph, sr = max(0.8, mu_t * 0.03), max(1.2, mu_h * 0.03), 0.2, max(2.5, mu_r * 0.04)
    ns = np.random.normal(mu_n, sn, samples_per_crop).clip(0, 140)
    ps = np.random.normal(mu_p, sp, samples_per_crop).clip(5, 145)
    ks = np.random.normal(mu_k, sk, samples_per_crop).clip(5, 205)
    ts = np.random.normal(mu_t, st, samples_per_crop).clip(8.8, 43.7)
    hs = np.random.normal(mu_h, sh, samples_per_crop).clip(14.0, 100.0)
    phs = np.random.normal(mu_ph, sph, samples_per_crop).clip(3.5, 9.9)
    rs = np.random.normal(mu_r, sr, samples_per_crop).clip(20.0, 298.6)
    for i in range(samples_per_crop):
        rows.append({
            'N': round(float(ns[i]), 2),
            'P': round(float(ps[i]), 2),
            'K': round(float(ks[i]), 2),
            'temperature': round(float(ts[i]), 2),
            'humidity': round(float(hs[i]), 2),
            'ph': round(float(phs[i]), 2),
            'rainfall': round(float(rs[i]), 2),
            'label': cname
        })

df = pd.DataFrame(rows)
df.to_csv('Crop_recommendation_1000.csv', index=False)
print('Done! Crop_recommendation_1000.csv created with', len(df), 'rows and', df['label'].nunique(), 'unique crops.')

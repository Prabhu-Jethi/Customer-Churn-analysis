# ChurnIQ Frontend

This package contains the **same visual direction as the original `churn_ui` HTML/CSS/JS**, upgraded with a fully interactive product preview and a matching Next.js implementation.

## Included

- `html-css-js/` — standalone HTML/CSS/JS version for Live Server.
- `nextjs/` — Next.js App Router version using the same design system.
- Responsive layout.
- Interactive workflow.
- Interactive customer risk controls.
- Low / Medium / High risk states:
  - Low: green
  - Medium: amber/yellow
  - High: red
- Reset.
- Run prediction demo.
- Product preview tabs.
- SHAP-style dynamic drivers.
- Retention recommendation modal.
- Mobile navigation.
- Toast feedback.
- Footer with email and author credit.

## Next.js

From the `nextjs` folder:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Important

The current probability calculation is a **frontend demo heuristic**. It is intentionally isolated in `scoreCustomer()` in `app/page.jsx`.

When the FastAPI backend is ready, replace that function with a request such as:

```text
POST /predict
```

and send the customer features to the real XGBoost model. The frontend UI does not need to be redesigned.


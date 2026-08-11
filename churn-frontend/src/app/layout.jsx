import "./globals.css";

export const metadata = {
  title: "ChurnIQ — Customer Retention Intelligence",
  description: "Customer churn prediction, explainability and retention intelligence."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

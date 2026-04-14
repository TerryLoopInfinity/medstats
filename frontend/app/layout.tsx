import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "MedStats — 临床统计分析平台",
  description: "在线临床医学统计分析工具",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <header className="border-b border-border sticky top-0 z-10 bg-background/95 backdrop-blur">
          <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-6">
            <Link href="/" className="font-bold text-lg tracking-tight">
              MedStats
            </Link>
            <nav className="flex gap-4 text-sm text-muted-foreground">
              <Link href="/upload" className="hover:text-foreground transition-colors">上传数据</Link>
              <Link href="/analyze" className="hover:text-foreground transition-colors">选择分析</Link>
              <Link href="/result" className="hover:text-foreground transition-colors">查看结果</Link>
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-border py-4 text-center text-xs text-muted-foreground">
          MedStats — 仅供科研参考，不构成临床决策依据
        </footer>
      </body>
    </html>
  );
}

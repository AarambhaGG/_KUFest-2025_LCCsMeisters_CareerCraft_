'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import {
  Zap,
  Menu,
  X,
  BarChart3,
  Target,
  BookOpen,
  MessageSquare,
  FileText,
  Sparkles,
  User,
} from "lucide-react"
import { ModeToggle } from '@/components/ToggleButton';
import { ProfileMenu } from '@/components/ProfileMenu';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const menuItems = [
    { label: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { label: 'Skill Analysis', href: '/dashboard/skills', icon: BarChart3 },
    { label: 'Job Match', href: '/dashboard/job-match', icon: Target },
    { label: 'Learning Roadmap', href: '/dashboard/roadmap', icon: BookOpen },
    { label: 'Interview Prep', href: '/dashboard/interview', icon: MessageSquare },
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 px-6 py-4 transition-colors">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition"
            >
              {sidebarOpen ? (
                <X className="h-5 w-5 text-slate-700 dark:text-slate-300" />
              ) : (
                <Menu className="h-5 w-5 text-slate-700 dark:text-slate-300" />
              )}
            </button>
            <Link href="/" className="flex items-center gap-2">
              <Zap className="h-8 w-8 text-purple-400" />
              <span className="text-xl font-bold text-slate-900 dark:text-white">
                CareerCraft
              </span>
            </Link>
          </div>

          <div className="flex items-center gap-4">
            <ModeToggle />
            <ProfileMenu userName="John Doe" userEmail="john@example.com" />
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside
          className={`${sidebarOpen ? "translate-x-0" : "-translate-x-full"
            } lg:translate-x-0 fixed lg:sticky lg:top-[60px] h-screen lg:h-[calc(100vh-100px)] w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 pt-6 transition-transform duration-300 z-30 overflow-y-auto`}
        >
          {/* <nav className="px-4 space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-3 px-4 py-3 rounded-lg text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white transition group"
                >
                  <Icon className="h-5 w-5 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition" />
                  <span>{item.label}</span>
                </Link>
              )
            })}
          </nav> */}

          {/* Dream Job Section */}
          <div className="px-4 border-t border-slate-200 dark:border-slate-800">
            {/* <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 px-4">
              Job Analysis
            </p> */}
            <div className="space-y-1">
              <Link
                href="/dashboard/dream-job-streaming"
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white transition group"
              >
                <Sparkles className="h-5 w-5 text-purple-600 dark:text-purple-400 group-hover:text-purple-600 dark:group-hover:text-purple-400" />
                <span>New Analysis</span>
              </Link>
              <Link
                href="/dashboard/my-analyses"
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white transition group"
              >
                <FileText className="h-5 w-5 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition" />
                <span>My Analyses</span>
              </Link>
              <Link
                href="/dashboard/ai-career-chat"
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white transition group"
              >
                <MessageSquare className="h-5 w-5 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition" />
                <span>AI Career Chat</span>
              </Link>
              <Link
                href="/dashboard/settings"
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white transition group"
              >
                <User className="h-5 w-5 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition" />
                <span>Profile</span>
              </Link>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1">
          <div className="p-6 lg:p-8">{children}</div>
        </main>
      </div>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-20"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  )
}

'use client';

import { ProfileChat } from '@/components/ProfileChat';

export default function AICareerChatPage() {
  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
          AI Career Chat
        </h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">
          Get personalized career advice and insights about your profile
        </p>
      </div>

      {/* Chat Component */}
      <ProfileChat />
    </div>
  );
}

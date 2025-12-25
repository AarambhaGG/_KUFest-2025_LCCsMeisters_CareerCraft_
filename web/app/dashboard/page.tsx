// 'use client';

// import Link from "next/link"
// import { Button } from '@/components/ui/button';
// import { TrendingUp, Award, AlertCircle, CheckCircle } from 'lucide-react';

// export default function Dashboard() {
//   return (
//     <div className="space-y-8">
//       {/* Header */}
//       {/* <div>
//         <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
//           Welcome to CareerCraft
//         </h1>
//         <p className="text-slate-600 dark:text-slate-400">
//           Your AI-powered career gap analysis and personalized roadmap to job
//           readiness
//         </p>
//       </div> */}

//       {/* Stats Grid */}
//       <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
//         {/* Job Match Score */}
//         <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-6 transition-colors">
//           <div className="flex items-center justify-between">
//             <div>
//               <p className="text-slate-600 dark:text-slate-400 text-sm font-medium">
//                 Dream Job Match
//               </p>
//               <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">
//                 72%
//               </p>
//             </div>
//             <TrendingUp className="h-8 w-8 text-purple-600 dark:text-purple-400" />
//           </div>
//           <p className="text-xs text-slate-500 dark:text-slate-500 mt-4">
//             Status: UNFIT → Working to FIT
//           </p>
//         </div>

//         {/* Current Skills */}
//         <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-6 transition-colors">
//           <div className="flex items-center justify-between">
//             <div>
//               <p className="text-slate-600 dark:text-slate-400 text-sm font-medium">
//                 Your Current Skills
//               </p>
//               <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">
//                 18
//               </p>
//             </div>
//             <Award className="h-8 w-8 text-blue-600 dark:text-blue-400" />
//           </div>
//           <p className="text-xs text-slate-500 dark:text-slate-500 mt-4">
//             Technical & Soft Skills
//           </p>
//         </div>

//         {/* Skill Gaps */}
//         <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-6 transition-colors">
//           <div className="flex items-center justify-between">
//             <div>
//               <p className="text-slate-600 dark:text-slate-400 text-sm font-medium">
//                 Critical Gaps
//               </p>
//               <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">
//                 6
//               </p>
//             </div>
//             <AlertCircle className="h-8 w-8 text-yellow-600 dark:text-yellow-400" />
//           </div>
//           <p className="text-xs text-slate-500 dark:text-slate-500 mt-4">
//             Skills to acquire for FIT
//           </p>
//         </div>

//         {/* Roadmap Progress */}
//         <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-6 transition-colors">
//           <div className="flex items-center justify-between">
//             <div>
//               <p className="text-slate-600 dark:text-slate-400 text-sm font-medium">
//                 Roadmap Progress
//               </p>
//               <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">
//                 28%
//               </p>
//             </div>
//             <CheckCircle className="h-8 w-8 text-green-600 dark:text-green-400" />
//           </div>
//           <p className="text-xs text-slate-500 dark:text-slate-500 mt-4">
//             10 weeks to go
//           </p>
//         </div>
//       </div>

//       {/* Quick Action Cards */}
//       <div className="grid md:grid-cols-2 gap-6">
//         {/* Profile Analysis */}
//         <div className="bg-gradient-to-br from-white to-slate-50 dark:from-slate-900 dark:to-slate-800 rounded-lg border border-slate-200 dark:border-purple-500/20 p-8 transition-colors">
//           <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-3">
//             My Profile Analysis
//           </h2>
//           <p className="text-slate-600 dark:text-slate-400 mb-6">
//             View your extracted skills, experience, and current readiness
//             assessment
//           </p>
//           <Button className="w-full bg-purple-600 hover:bg-purple-700">
//             View Profile
//           </Button>
//         </div>

//         {/* Dream Job Match */}
//         <div className="bg-gradient-to-br from-white to-slate-50 dark:from-slate-900 dark:to-slate-800 rounded-lg border border-slate-200 dark:border-purple-500/20 p-8 transition-colors">
//           <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-3">
//             Dream Job Assessment
//           </h2>
//           <p className="text-slate-600 dark:text-slate-400 mb-6">
//             Paste a job description and get your FIT/UNFIT assessment with gap
//             analysis
//           </p>
//           <Link href="/dashboard/dream-job">
//             <Button className="w-full bg-purple-600 hover:bg-purple-700">
//               Analyze Job
//             </Button>
//           </Link>
//         </div>
//       </div>

//       {/* Top Skills Section */}
//       <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-8 transition-colors">
//         <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-6">
//           Your Current Skills
//         </h2>
//         <div className="space-y-4">
//           {[
//             { name: "React.js", level: 92, color: "bg-blue-500" },
//             { name: "TypeScript", level: 85, color: "bg-blue-400" },
//             { name: "Next.js", level: 88, color: "bg-purple-500" },
//             { name: "Tailwind CSS", level: 90, color: "bg-cyan-500" },
//             { name: "JavaScript", level: 95, color: "bg-yellow-500" },
//           ].map((skill) => (
//             <div key={skill.name} className="flex items-center gap-4">
//               <div className="w-24 flex-shrink-0">
//                 <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
//                   {skill.name}
//                 </p>
//               </div>
//               <div className="flex-1 h-2 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
//                 <div
//                   className={`h-full ${skill.color} transition-all duration-500`}
//                   style={{ width: `${skill.level}%` }}
//                 />
//               </div>
//               <p className="text-sm font-semibold text-slate-700 dark:text-slate-300 w-12 text-right">
//                 {skill.level}%
//               </p>
//             </div>
//           ))}
//         </div>
//       </div>
//     </div>
//   )
// }

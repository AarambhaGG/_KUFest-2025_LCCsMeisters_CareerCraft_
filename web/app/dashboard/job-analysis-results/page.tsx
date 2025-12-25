'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  ArrowLeft,
  CheckCircle,
  XCircle,
  AlertCircle,
  TrendingUp,
  Target,
  Clock,
  Lightbulb,
  Award,
  BookOpen,
  Sparkles,
  Briefcase,
  GraduationCap,
} from 'lucide-react';

interface JobAnalysisResult {
  id: number;
  user: number;
  user_email: string;
  user_name: string;
  job: number;
  job_title: string;
  job_company: string;
  additional_context: string;
  eligibility_level: 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR';
  match_score: number;
  analysis_summary: string;
  strengths: string;
  gaps: string;
  recommendations: string;
  matching_skills: string;
  missing_skills: string;
  skill_gaps: string;
  skills_match_score: number;
  experience_match_score: number;
  education_match_score: number;
  culture_fit_score: number;
  location_match_score: number;
  salary_match_score: number;
  technical_skills_score: number;
  soft_skills_score: number;
  domain_knowledge_score: number;
  experience_match: string;
  experience_gap_years: string;
  years_of_experience_required: string;
  years_of_experience_user: string;
  readiness_percentage: number;
  estimated_preparation_time: string;
  confidence_level: 'VERY_HIGH' | 'HIGH' | 'MEDIUM' | 'LOW';
  next_steps: string;
  priority_improvements: string;
  learning_resources: string;
  analyzed_at: string;
  llm_model: string;
  full_analysis: string;
  token_usage: number;
}

export default function JobAnalysisResultsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [analysisData, setAnalysisData] = useState<JobAnalysisResult | null>(null);

  useEffect(() => {
    const data = searchParams.get('data');
    if (data) {
      try {
        const parsed = JSON.parse(decodeURIComponent(data));
        setAnalysisData(parsed);
      } catch (error) {
        console.error('Failed to parse analysis data:', error);
      }
    }
  }, [searchParams]);

  if (!analysisData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-slate-600 dark:text-slate-400">Loading analysis results...</p>
        </div>
      </div>
    );
  }

  const getEligibilityColor = (level: string) => {
    switch (level) {
      case 'EXCELLENT':
        return 'bg-green-100 dark:bg-green-950 text-green-700 dark:text-green-300 border-green-200 dark:border-green-800';
      case 'GOOD':
        return 'bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800';
      case 'FAIR':
        return 'bg-yellow-100 dark:bg-yellow-950 text-yellow-700 dark:text-yellow-300 border-yellow-200 dark:border-yellow-800';
      case 'POOR':
        return 'bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-300 border-red-200 dark:border-red-800';
      default:
        return 'bg-slate-100 dark:bg-slate-950 text-slate-700 dark:text-slate-300';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400';
    if (score >= 60) return 'text-blue-600 dark:text-blue-400';
    if (score >= 40) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getProgressColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-blue-500';
    if (score >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const parseListString = (str: string): string[] => {
    if (!str) return [];
    return str.split('\n').filter((item) => item.trim());
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link href="/dashboard/dream-job">
          <Button variant="ghost" size="sm" className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </Link>
      </div>

      {/* Title Section */}
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-8">
          <Badge
            className={`mb-4 ${getEligibilityColor(
              analysisData.eligibility_level
            )}`}
          >
            <Award className="h-3 w-3 mr-1" />
            {analysisData.eligibility_level} Match
          </Badge>
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-3">
            Job Analysis Results
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            {analysisData.job_title}
            {analysisData.job_company && ` at ${analysisData.job_company}`}
          </p>
        </div>

        {/* Overall Score Card */}
        <Card className="border-purple-100 dark:border-purple-900/50 mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-purple-600" />
              Overall Match Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-6">
              <div className="flex-1">
                <div className="flex items-baseline gap-2 mb-2">
                  <span
                    className={`text-5xl font-bold ${getScoreColor(
                      analysisData.match_score
                    )}`}
                  >
                    {analysisData.match_score}%
                  </span>
                  <span className="text-slate-600 dark:text-slate-400">
                    match
                  </span>
                </div>
                <Progress value={analysisData.match_score} className="h-3" />
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {analysisData.readiness_percentage}%
                </p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Ready
                </p>
              </div>
            </div>
            {analysisData.estimated_preparation_time && (
              <div className="mt-4 flex items-center gap-2 text-slate-600 dark:text-slate-400">
                <Clock className="h-4 w-4" />
                <span className="text-sm">
                  Estimated preparation time:{" "}
                  <strong>{analysisData.estimated_preparation_time}</strong>
                </span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Summary */}
        {analysisData.analysis_summary && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-purple-600" />
                Analysis Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-slate-700 dark:text-slate-300 whitespace-pre-line">
                {analysisData.analysis_summary}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Detailed Scores Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Skills Score */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Skills Match</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Overall Skills
                    </span>
                    <span
                      className={`text-sm font-semibold ${getScoreColor(
                        analysisData.skills_match_score
                      )}`}
                    >
                      {analysisData.skills_match_score}%
                    </span>
                  </div>
                  <Progress
                    value={analysisData.skills_match_score}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Technical Skills
                    </span>
                    <span
                      className={`text-sm font-semibold ${getScoreColor(
                        analysisData.technical_skills_score
                      )}`}
                    >
                      {analysisData.technical_skills_score}%
                    </span>
                  </div>
                  <Progress
                    value={analysisData.technical_skills_score}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Soft Skills
                    </span>
                    <span
                      className={`text-sm font-semibold ${getScoreColor(
                        analysisData.soft_skills_score
                      )}`}
                    >
                      {analysisData.soft_skills_score}%
                    </span>
                  </div>
                  <Progress
                    value={analysisData.soft_skills_score}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Domain Knowledge
                    </span>
                    <span
                      className={`text-sm font-semibold ${getScoreColor(
                        analysisData.domain_knowledge_score
                      )}`}
                    >
                      {analysisData.domain_knowledge_score}%
                    </span>
                  </div>
                  <Progress
                    value={analysisData.domain_knowledge_score}
                    className="h-2"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Other Scores */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Other Factors</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Experience
                    </span>
                    <span
                      className={`text-sm font-semibold ${getScoreColor(
                        analysisData.experience_match_score
                      )}`}
                    >
                      {analysisData.experience_match_score}%
                    </span>
                  </div>
                  <Progress
                    value={analysisData.experience_match_score}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Education
                    </span>
                    <span
                      className={`text-sm font-semibold ${getScoreColor(
                        analysisData.education_match_score
                      )}`}
                    >
                      {analysisData.education_match_score}%
                    </span>
                  </div>
                  <Progress
                    value={analysisData.education_match_score}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Culture Fit
                    </span>
                    <span
                      className={`text-sm font-semibold ${getScoreColor(
                        analysisData.culture_fit_score
                      )}`}
                    >
                      {analysisData.culture_fit_score}%
                    </span>
                  </div>
                  <Progress
                    value={analysisData.culture_fit_score}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      Confidence Level
                    </span>
                    <span className="text-sm font-semibold text-purple-600 dark:text-purple-400">
                      {analysisData.confidence_level}
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Strengths & Gaps */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Strengths */}
          {analysisData.strengths && (
            <Card className="border-green-100 dark:border-green-900/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-green-700 dark:text-green-300">
                  <CheckCircle className="h-5 w-5" />
                  Your Strengths
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {parseListString(analysisData.strengths).map(
                    (strength, idx) => (
                      <li
                        key={idx}
                        className="flex items-start gap-2 text-slate-700 dark:text-slate-300"
                      >
                        <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400 shrink-0 mt-0.5" />
                        <span className="text-sm">{strength}</span>
                      </li>
                    )
                  )}
                </ul>
              </CardContent>
            </Card>
          )}

          {/* Gaps */}
          {analysisData.gaps && (
            <Card className="border-red-100 dark:border-red-900/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-red-700 dark:text-red-300">
                  <XCircle className="h-5 w-5" />
                  Areas to Improve
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {parseListString(analysisData.gaps).map((gap, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-2 text-slate-700 dark:text-slate-300"
                    >
                      <XCircle className="h-4 w-4 text-red-600 dark:text-red-400 shrink-0 mt-0.5" />
                      <span className="text-sm">{gap}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Skills Breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Matching Skills */}
          {analysisData.matching_skills && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  Matching Skills
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {parseListString(analysisData.matching_skills).map(
                    (skill, idx) => (
                      <Badge
                        key={idx}
                        className="bg-green-100 dark:bg-green-950 text-green-700 dark:text-green-300"
                      >
                        {skill}
                      </Badge>
                    )
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Missing Skills */}
          {analysisData.missing_skills && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-red-600" />
                  Skills to Acquire
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {parseListString(analysisData.missing_skills).map(
                    (skill, idx) => (
                      <Badge
                        key={idx}
                        className="bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-300"
                      >
                        {skill}
                      </Badge>
                    )
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Recommendations */}
        {analysisData.recommendations && (
          <Card className="border-blue-100 dark:border-blue-900/50 mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
                <Lightbulb className="h-5 w-5" />
                Recommendations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                {parseListString(analysisData.recommendations).map(
                  (rec, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-2 text-slate-700 dark:text-slate-300"
                    >
                      <span className="text-blue-600 dark:text-blue-400 font-semibold shrink-0">
                        {idx + 1}.
                      </span>
                      <span>{rec}</span>
                    </li>
                  )
                )}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* Next Steps */}
        {analysisData.next_steps && (
          <Card className="border-purple-100 dark:border-purple-900/50 mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-purple-700 dark:text-purple-300">
                <TrendingUp className="h-5 w-5" />
                Next Steps
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {parseListString(analysisData.next_steps).map((step, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <div className="flex items-center justify-center w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 font-semibold text-sm shrink-0">
                      {idx + 1}
                    </div>
                    <p className="text-slate-700 dark:text-slate-300 pt-0.5">
                      {step}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Priority Improvements */}
        {analysisData.priority_improvements && (
          <Card className="border-orange-100 dark:border-orange-900/50 mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-orange-700 dark:text-orange-300">
                <Target className="h-5 w-5" />
                Priority Improvements
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {parseListString(analysisData.priority_improvements).map(
                  (improvement, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-2 text-slate-700 dark:text-slate-300"
                    >
                      <span className="text-orange-600 dark:text-orange-400 font-semibold shrink-0">
                        •
                      </span>
                      <span>{improvement}</span>
                    </li>
                  )
                )}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* Learning Resources */}
        {analysisData.learning_resources && (
          <Card className="border-indigo-100 dark:border-indigo-900/50 mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-indigo-700 dark:text-indigo-300">
                <BookOpen className="h-5 w-5" />
                Learning Resources
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-slate-700 dark:text-slate-300 whitespace-pre-line">
                {analysisData.learning_resources}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Action Buttons */}
        <div className="flex gap-4 justify-center">
          <Link href="/dashboard/dream-job-streaming">
            <Button variant="outline" size="lg">
              Back to Dashboard
            </Button>
          </Link>
          <Link href="/dashboard/dream-job">
            <Button size="lg" className="bg-purple-600 hover:bg-purple-700">
              <Briefcase className="mr-2 h-5 w-5" />
              Analyze Another Job
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}

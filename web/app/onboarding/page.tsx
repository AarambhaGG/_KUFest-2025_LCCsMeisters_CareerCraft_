'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Zap, Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import { ModeToggle } from '@/components/ToggleButton';
import { ProfileMenu } from '@/components/ProfileMenu';
import { uploadResumeOnboard } from '@/api';

export default function Onboarding() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const fileExtension = file.name.split('.').pop()?.toLowerCase();
      if (fileExtension !== 'pdf' && fileExtension !== 'docx') {
        setError('Please upload only PDF or DOCX files');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select a file to upload');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const response = await uploadResumeOnboard(selectedFile);
      console.log('Profile created:', response);
      setUploadSuccess(true);

      setTimeout(() => {
        router.push("/dashboard/dream-job-streaming")
      }, 1500);
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(err.response?.data?.message || 'Failed to upload resume. Please try again.');
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-50 via-purple-50 to-slate-50 dark:from-slate-900 dark:via-purple-900 dark:to-slate-900 transition-colors">
      {/* Navigation */}
      <nav className="flex items-center justify-between px-6 py-4 sm:px-12 border-b border-slate-200 dark:border-purple-500/20">
        <Link href="/" className="flex items-center gap-2">
          <Zap className="h-8 w-8 text-purple-600 dark:text-purple-400" />
          <span className="text-2xl font-bold text-slate-900 dark:text-white">
            CareerCraft
          </span>
        </Link>
        <div className="flex items-center gap-4">
          <ModeToggle />
          <ProfileMenu userName="John Doe" userEmail="john@example.com" />
        </div>
      </nav>

      <div className="flex items-center justify-center py-12 px-4 sm:px-6">
        <div className="w-full max-w-2xl">
          <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-purple-500/20 p-8 sm:p-10 transition-colors">
            {/* Header */}
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-100 dark:bg-purple-900/30 rounded-full mb-4">
                <Upload className="h-8 w-8 text-purple-600 dark:text-purple-400" />
              </div>
              <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
                Upload Your Resume
              </h2>
              <p className="text-slate-600 dark:text-slate-400">
                Upload your CV in PDF or DOCX format to build your profile
                automatically
              </p>
            </div>

            {/* Upload Area */}
            <div className="space-y-6">
              <div className="border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-lg p-8 text-center hover:border-purple-500 dark:hover:border-purple-500 transition-colors">
                <input
                  type="file"
                  id="resume-upload"
                  accept=".pdf,.docx"
                  onChange={handleFileChange}
                  className="hidden"
                  disabled={isUploading}
                />
                <label
                  htmlFor="resume-upload"
                  className="cursor-pointer flex flex-col items-center"
                >
                  <FileText className="h-12 w-12 text-slate-400 dark:text-slate-500 mb-4" />
                  <p className="text-lg font-medium text-slate-900 dark:text-white mb-2">
                    {selectedFile
                      ? selectedFile.name
                      : "Choose a file or drag it here"}
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-400">
                    Supported formats: PDF, DOCX (Max 10MB)
                  </p>
                </label>
              </div>

              {/* Selected File Info */}
              {selectedFile && (
                <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-500/30 rounded-lg p-4 flex items-center gap-3">
                  <FileText className="h-5 w-5 text-purple-600 dark:text-purple-400 shrink-0" />
                  <div className="flex-1">
                    <p className="font-medium text-purple-900 dark:text-purple-100">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-purple-700 dark:text-purple-300">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedFile(null)}
                    disabled={isUploading}
                    className="border-purple-300 dark:border-purple-700 text-purple-600 dark:text-purple-400"
                  >
                    Remove
                  </Button>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/30 rounded-lg p-4 flex gap-3">
                  <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-700 dark:text-red-300">
                      Upload Failed
                    </p>
                    <p className="text-sm text-red-600 dark:text-red-400">
                      {error}
                    </p>
                  </div>
                </div>
              )}

              {/* Success Message */}
              {uploadSuccess && (
                <div className="bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/30 rounded-lg p-4 flex gap-3">
                  <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-green-700 dark:text-green-300">
                      Profile Created Successfully!
                    </p>
                    <p className="text-sm text-green-600 dark:text-green-400">
                      Redirecting to your dashboard...
                    </p>
                  </div>
                </div>
              )}

              {/* Upload Button */}
              <Button
                onClick={handleSubmit}
                disabled={!selectedFile || isUploading || uploadSuccess}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-slate-400 dark:disabled:bg-slate-700"
              >
                {isUploading ? (
                  <>
                    <span className="animate-spin mr-2">⏳</span>
                    Processing Resume...
                  </>
                ) : uploadSuccess ? (
                  <>
                    <CheckCircle className="mr-2 h-5 w-5" />
                    Complete!
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-5 w-5" />
                    Build My Profile
                  </>
                )}
              </Button>

              {/* Skip Button */}
              <Button
                onClick={() => router.push("/dashboard/dream-job-streaming")}
                disabled={isUploading || uploadSuccess}
                variant="outline"
                className="w-full border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                Skip for Now
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

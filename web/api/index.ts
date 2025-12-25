import axiosInstance from "@/services/axios";

// Auth APIs
export const loginUser = async (email: string, password: string) => {
  const { data } = await axiosInstance.post("/users/auth/login/", { email, password });
  return data;
};

export const signupUser = async (userData: {
  username: string;
  email: string;
  password: string;
  password_confirm: string;
  first_name: string;
  last_name: string;
}) => {
  const { data } = await axiosInstance.post("/users/auth/register/", userData);
  return data;
};

export const logoutUser = async () => {
  const { data } = await axiosInstance.post("/auth/logout/");
  return data;
};

export const getCurrentUser = async () => {
  const { data } = await axiosInstance.get("/auth/me/");
  return data;
};

// File Upload APIs
export const uploadFiles = async (files: File[]) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const { data } = await axiosInstance.post("/upload/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

// Skills APIs
export const getSkills = async () => {
  const { data } = await axiosInstance.get("/skills/");
  return data;
};

export const getUserSkills = async () => {
  const { data } = await axiosInstance.get("/skills/user/");
  return data;
};

export const addUserSkill = async (skillData: {
  skill_id: string;
  proficiency_level: string;
}) => {
  const { data } = await axiosInstance.post("/skills/user/", skillData);
  return data;
};

// Job Match APIs
export const analyzeJobMatch = async (jobUrl: string) => {
  const { data } = await axiosInstance.post("/job-match/analyze/", {
    job_url: jobUrl,
  });
  return data;
};

// Roadmap APIs
export const getRoadmap = async () => {
  const { data } = await axiosInstance.get("/roadmap/");
  return data;
};

export const updateRoadmapProgress = async (weekId: string) => {
  const { data } = await axiosInstance.patch(`/roadmap/${weekId}/`, {
    completed: true,
  });
  return data;
};

// User Profile APIs (for basic user info)
export const getUserProfile = async () => {
  const { data } = await axiosInstance.get("/users/profile/");
  return data;
};

export const updateUserProfile = async (profileData: {
  username?: string;
  first_name?: string;
  last_name?: string;
  phone_number?: string;
  date_of_birth?: string;
}) => {
  const { data } = await axiosInstance.patch("/users/profile/", profileData);
  return data;
};

// Onboarding API - Upload resume and build profile
export const uploadResumeOnboard = async (resumeFile: File) => {
  const formData = new FormData();
  formData.append("resume", resumeFile);

  console.log("Uploading resume:", {
    name: resumeFile.name,
    type: resumeFile.type,
    size: resumeFile.size,
  });

  // Extended timeout for AI processing (5 minutes)
  const { data } = await axiosInstance.post("/profiles/profile/onboard/", formData, {
    timeout: 300000, // 5 minutes
  });
  return data;
};

// Job Analysis APIs
export const analyzeDreamJob = async (jobData: {
  job_description: string;
  additional_context?: string;
  save_job?: boolean;
}) => {
  // Extended timeout for AI processing (5 minutes)
  const { data } = await axiosInstance.post("/jobs/analyses/analyze_dream_job/", jobData, {
    timeout: 300000, // 5 minutes
  });
  return data;
};

// Education APIs
export const getEducation = async () => {
  const { data } = await axiosInstance.get("/profiles/education/");
  return data;
};

export const createEducation = async (educationData: {
  institution: string;
  degree: string;
  degree_level: string;
  field_of_study?: string;
  start_date: string;
  end_date?: string;
  is_current: boolean;
  grade?: string;
  description?: string;
}) => {
  const { data } = await axiosInstance.post("/profiles/education/", educationData);
  return data;
};

export const updateEducation = async (id: string, educationData: Partial<{
  institution: string;
  degree: string;
  degree_level: string;
  field_of_study?: string;
  start_date: string;
  end_date?: string;
  is_current: boolean;
  grade?: string;
  description?: string;
}>) => {
  const { data } = await axiosInstance.patch(`/profiles/education/${id}/`, educationData);
  return data;
};

export const deleteEducation = async (id: string) => {
  const { data } = await axiosInstance.delete(`/profiles/education/${id}/`);
  return data;
};

// Work Experience APIs
export const getWorkExperience = async () => {
  const { data } = await axiosInstance.get("/profiles/work-experiences/");
  return data;
};

export const createWorkExperience = async (experienceData: {
  job_title: string;
  company: string;
  employment_type: string;
  location?: string;
  is_remote: boolean;
  start_date: string;
  end_date?: string;
  is_current: boolean;
  description?: string;
}) => {
  const { data } = await axiosInstance.post("/profiles/work-experiences/", experienceData);
  return data;
};

export const updateWorkExperience = async (id: string, experienceData: Partial<{
  job_title: string;
  company: string;
  employment_type: string;
  location?: string;
  is_remote: boolean;
  start_date: string;
  end_date?: string;
  is_current: boolean;
  description?: string;
}>) => {
  const { data } = await axiosInstance.patch(`/profiles/work-experiences/${id}/`, experienceData);
  return data;
};

export const deleteWorkExperience = async (id: string) => {
  const { data } = await axiosInstance.delete(`/profiles/work-experiences/${id}/`);
  return data;
};

// Projects APIs
export const getProjects = async () => {
  const { data } = await axiosInstance.get("/profiles/projects/");
  return data;
};

export const createProject = async (projectData: {
  title: string;
  project_type: string;
  description?: string;
  technologies_used?: string[];
  project_url?: string;
  github_url?: string;
  demo_url?: string;
  start_date: string;
  end_date?: string;
  is_ongoing: boolean;
}) => {
  const { data } = await axiosInstance.post("/profiles/projects/", projectData);
  return data;
};

export const updateProject = async (id: string, projectData: Partial<{
  title: string;
  project_type: string;
  description?: string;
  technologies_used?: string[];
  project_url?: string;
  github_url?: string;
  demo_url?: string;
  start_date: string;
  end_date?: string;
  is_ongoing: boolean;
}>) => {
  const { data } = await axiosInstance.patch(`/profiles/projects/${id}/`, projectData);
  return data;
};

export const deleteProject = async (id: string) => {
  const { data } = await axiosInstance.delete(`/profiles/projects/${id}/`);
  return data;
};

// Certifications APIs
export const getCertifications = async () => {
  const { data } = await axiosInstance.get("/profiles/certifications/");
  return data;
};

export const createCertification = async (certificationData: {
  name: string;
  issuing_organization: string;
  credential_id?: string;
  credential_url?: string;
  issue_date: string;
  expiry_date?: string;
  does_not_expire: boolean;
}) => {
  const { data } = await axiosInstance.post("/profiles/certifications/", certificationData);
  return data;
};

export const updateCertification = async (id: string, certificationData: Partial<{
  name: string;
  issuing_organization: string;
  credential_id?: string;
  credential_url?: string;
  issue_date: string;
  expiry_date?: string;
  does_not_expire: boolean;
}>) => {
  const { data } = await axiosInstance.patch(`/profiles/certifications/${id}/`, certificationData);
  return data;
};

export const deleteCertification = async (id: string) => {
  const { data } = await axiosInstance.delete(`/profiles/certifications/${id}/`);
  return data;
};

// Profile Chat API
export const chatAboutProfile = async (message: string) => {
  const { data } = await axiosInstance.post("/profiles/profile/chat/", { message }, {
    timeout: 120000, // 2 minutes for AI processing
  });
  return data;
};

// Analysis Chat API
export const chatAboutAnalysis = async (analysisId: number, message: string) => {
  const { data } = await axiosInstance.post("/jobs/analyses/chat/", {
    analysis_id: analysisId,
    message
  }, {
    timeout: 120000, // 2 minutes for AI processing
  });
  return data;
};

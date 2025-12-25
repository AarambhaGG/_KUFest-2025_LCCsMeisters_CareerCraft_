"""
AI-powered services for profile management
"""

import json
import os
from typing import Dict, Any, Optional
import PyPDF2
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from django.core.files.uploadedfile import UploadedFile
from langchain_google_genai import ChatGoogleGenerativeAI


class ResumeParserService:
    """
    Service for parsing resumes using GPT-4 to extract structured data
    """

    def __init__(self):
        """Initialize the dream job parser with GPT-4"""
        model_provider = os.getenv("MODEL_PROVIDER")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        gemini_model = "gemini-3-flash-preview"
        openai_model = "gpt-4o-mini"

        if model_provider == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            self.llm = ChatGoogleGenerativeAI(model=gemini_model, api_key=gemini_api_key)
        else:
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=0.1,  # Low temperature for consistent extraction
                api_key=api_key,
            )

    def extract_text_from_pdf(self, file: UploadedFile) -> str:
        """
        Extract text content from PDF file

        Args:
            file: Uploaded PDF file

        Returns:
            Extracted text content
        """
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    def extract_text_from_docx(self, file: UploadedFile) -> str:
        """
        Extract text content from DOCX file

        Args:
            file: Uploaded DOCX file

        Returns:
            Extracted text content
        """
        try:
            doc = Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from DOCX: {str(e)}")

    def extract_text(self, file: UploadedFile) -> str:
        """
        Extract text from resume file (PDF or DOCX)

        Args:
            file: Uploaded resume file

        Returns:
            Extracted text content
        """
        file_extension = file.name.lower().split(".")[-1]

        if file_extension == "pdf":
            return self.extract_text_from_pdf(file)
        elif file_extension in ["docx", "doc"]:
            return self.extract_text_from_docx(file)
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                "Please upload a PDF or DOCX file."
            )

    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse resume text using GPT-4 to extract structured data

        Args:
            resume_text: Extracted text from resume

        Returns:
            Structured dictionary with parsed resume data
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert resume parser. Extract structured information from resumes.

Your task is to analyze the resume text and extract the following information in JSON format:

{{
  "personal_info": {{
    "name": "Full name",
    "email": "Email address",
    "phone": "Phone number",
    "location": "City, State/Country",
    "linkedin": "LinkedIn URL (if available)",
    "github": "GitHub URL (if available)",
    "portfolio": "Portfolio URL (if available)"
  }},
  "summary": "Professional summary or objective",
  "skills": [
    {{
      "name": "Skill name",
      "category": "TECHNICAL/SOFT/TOOL/LANGUAGE",
      "proficiency": "BEGINNER/INTERMEDIATE/ADVANCED/EXPERT",
      "years_of_experience": <estimated years as float>
    }}
  ],
  "work_experience": [
    {{
      "job_title": "Title",
      "company": "Company name",
      "location": "City, State",
      "employment_type": "FULL_TIME/PART_TIME/CONTRACT/FREELANCE/INTERNSHIP",
      "is_remote": <true/false>,
      "start_date": "YYYY-MM" or "YYYY-MM-DD",
      "end_date": "YYYY-MM" or "YYYY-MM-DD" or null,
      "is_current": <true/false>,
      "description": "Brief description",
      "responsibilities": ["Responsibility 1", "Responsibility 2"],
      "achievements": ["Achievement 1", "Achievement 2"],
      "skills_used": ["Skill 1", "Skill 2"]
    }}
  ],
  "education": [
    {{
      "institution": "University/School name",
      "degree": "Degree name",
      "degree_level": "HIGH_SCHOOL/ASSOCIATE/BACHELOR/MASTER/PHD/CERTIFICATE/BOOTCAMP",
      "field_of_study": "Major/Field",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM" or null,
      "is_current": <true/false>,
      "grade": "GPA or grade",
      "honors": ["Honor 1", "Honor 2"]
    }}
  ],
  "projects": [
    {{
      "title": "Project name",
      "project_type": "PERSONAL/WORK/ACADEMIC/OPEN_SOURCE",
      "description": "Project description",
      "technologies_used": ["Tech 1", "Tech 2"],
      "project_url": "URL (if available)",
      "github_url": "GitHub URL (if available)",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM" or null,
      "is_ongoing": <true/false>,
      "skills_demonstrated": ["Skill 1", "Skill 2"]
    }}
  ],
  "certifications": [
    {{
      "name": "Certification name",
      "issuing_organization": "Organization",
      "credential_id": "Credential ID (if available)",
      "credential_url": "Verification URL (if available)",
      "issue_date": "YYYY-MM",
      "expiry_date": "YYYY-MM" or null,
      "does_not_expire": <true/false>,
      "skills_validated": ["Skill 1", "Skill 2"]
    }}
  ],
  "total_years_experience": <calculated total as float>,
  "current_title": "Most recent job title",
  "current_company": "Most recent company",
  "career_level": "ENTRY/JUNIOR/MID/SENIOR/LEAD/EXECUTIVE"
}}

IMPORTANT RULES:
1. Extract ALL information available, even if incomplete
2. For missing fields, use null or empty arrays
3. Infer proficiency levels from context (years of experience, job titles, project complexity)
4. Calculate years of experience for skills based on work history
5. Be conservative with proficiency estimates
6. Extract skills from work descriptions, projects, and certifications
7. Standardize skill names (e.g., "JavaScript" not "javascript", "React.js" not "React")
8. For dates, use YYYY-MM format; use YYYY-MM-DD if day is specified
9. Categorize skills accurately (TECHNICAL for programming, SOFT for communication, etc.)
10. Return ONLY valid JSON, no additional text

If information is ambiguous or missing, make reasonable inferences based on context.""",
                ),
                ("human", "Resume text:\n\n{resume_text}"),
            ]
        )

        try:
            # Invoke GPT-4 to parse resume
            chain = prompt | self.llm
            response = chain.invoke({"resume_text": resume_text})

            # Extract JSON from response
            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove ```

            # Parse JSON
            parsed_data = json.loads(content.strip())

            return parsed_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GPT-4 response as JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing resume with GPT-4: {str(e)}")

    def parse_resume_file(self, file: UploadedFile) -> Dict[str, Any]:
        """
        Complete resume parsing pipeline: extract text + parse with GPT-4

        Args:
            file: Uploaded resume file (PDF or DOCX)

        Returns:
            Dictionary containing:
                - resume_text: Extracted text
                - parsed_data: Structured data from GPT-4
        """
        # Step 1: Extract text
        resume_text = self.extract_text(file)

        if not resume_text or len(resume_text.strip()) < 50:
            raise ValueError(
                "Resume appears to be empty or too short. "
                "Please ensure the file contains readable text."
            )

        # Step 2: Parse with GPT-4
        parsed_data = self.parse_resume(resume_text)

        return {
            "resume_text": resume_text,
            "parsed_data": parsed_data,
        }


class ProfileBuilderService:
    """
    Service for building user profiles from parsed resume data
    """

    def __init__(self, user, parsed_data: Dict[str, Any]):
        """
        Initialize profile builder

        Args:
            user: User instance
            parsed_data: Parsed resume data from ResumeParserService
        """
        self.user = user
        self.parsed_data = parsed_data

    def create_or_update_profile(self, resume_file=None, resume_text: str = None) -> Any:
        """
        Create or update user profile from parsed resume data

        Args:
            resume_file: Uploaded resume file (optional)
            resume_text: Extracted resume text

        Returns:
            UserProfile instance
        """
        from apps.profiles.models import UserProfile

        profile_data = self.parsed_data.get("personal_info", {})

        # Prepare URLs, handle None values safely
        linkedin_url = (profile_data.get("linkedin") or "").strip() if profile_data.get("linkedin") else ""
        github_url = (profile_data.get("github") or "").strip() if profile_data.get("github") else ""
        portfolio_url = (profile_data.get("portfolio") or "").strip() if profile_data.get("portfolio") else ""

        profile, created = UserProfile.objects.get_or_create(
            user=self.user,
            defaults={
                "bio": self.parsed_data.get("summary", ""),
                "current_title": self.parsed_data.get("current_title", ""),
                "current_company": self.parsed_data.get("current_company", ""),
                "years_of_experience": self.parsed_data.get("total_years_experience", 0),
                "linkedin_url": linkedin_url,
                "github_url": github_url,
                "portfolio_url": portfolio_url,
                "resume_text": resume_text or "",
                "resume_parsed_data": self.parsed_data,
            },
        )

        if not created:
            # Update existing profile
            profile.bio = self.parsed_data.get("summary", profile.bio)
            profile.current_title = self.parsed_data.get("current_title", profile.current_title)
            profile.current_company = self.parsed_data.get("current_company", profile.current_company)
            profile.years_of_experience = self.parsed_data.get(
                "total_years_experience", profile.years_of_experience
            )
            # Update URLs only if new values are provided
            if linkedin_url:
                profile.linkedin_url = linkedin_url
            if github_url:
                profile.github_url = github_url
            if portfolio_url:
                profile.portfolio_url = portfolio_url
            profile.resume_text = resume_text or profile.resume_text
            profile.resume_parsed_data = self.parsed_data

        if resume_file:
            profile.resume = resume_file

        profile.save()
        return profile

    def _parse_date(self, date_str: str) -> str:
        """
        Parse date string and convert to YYYY-MM-DD format

        Args:
            date_str: Date string in YYYY, YYYY-MM or YYYY-MM-DD format

        Returns:
            Date string in YYYY-MM-DD format
        """
        if not date_str:
            return None

        # Convert to string and strip whitespace
        date_str = str(date_str).strip()

        # If already in YYYY-MM-DD format, return as is
        if len(date_str) == 10:
            return date_str

        # If in YYYY-MM format, append -01
        if len(date_str) == 7:
            return f"{date_str}-01"

        # If in YYYY format (just year), append -01-01
        if len(date_str) == 4 and date_str.isdigit():
            return f"{date_str}-01-01"

        # Default fallback
        return date_str

    def create_education_records(self, profile) -> list:
        """
        Create education records from parsed data

        Args:
            profile: UserProfile instance

        Returns:
            List of created Education instances
        """
        from apps.profiles.models import Education

        education_records = []
        for edu_data in self.parsed_data.get("education", []):
            try:
                education = Education.objects.create(
                    profile=profile,
                    institution=edu_data.get("institution", ""),
                    degree=edu_data.get("degree", ""),
                    degree_level=edu_data.get("degree_level", "BACHELOR"),
                    field_of_study=edu_data.get("field_of_study", ""),
                    start_date=self._parse_date(edu_data.get("start_date")) or "2000-01-01",
                    end_date=self._parse_date(edu_data.get("end_date")),
                    is_current=edu_data.get("is_current", False),
                    grade=edu_data.get("grade", ""),
                )
                education_records.append(education)
            except Exception as e:
                # Log error but continue with other records
                print(f"Error creating education record: {str(e)}")
                continue

        return education_records

    def create_work_experience_records(self, profile) -> list:
        """
        Create work experience records from parsed data

        Args:
            profile: UserProfile instance

        Returns:
            List of created WorkExperience instances
        """
        from apps.profiles.models import WorkExperience, Skill, SkillCategory

        work_records = []

        # Get or create default category
        default_category, _ = SkillCategory.objects.get_or_create(
            name="General",
            defaults={"description": "General skills"},
        )

        for work_data in self.parsed_data.get("work_experience", []):
            try:
                # Ensure location is never None
                location = work_data.get("location") or ""
                if location is None:
                    location = ""

                work_exp = WorkExperience.objects.create(
                    profile=profile,
                    job_title=work_data.get("job_title", ""),
                    company=work_data.get("company", ""),
                    employment_type=work_data.get("employment_type", "FULL_TIME"),
                    location=location,
                    is_remote=work_data.get("is_remote", False),
                    start_date=self._parse_date(work_data.get("start_date")) or "2000-01-01",
                    end_date=self._parse_date(work_data.get("end_date")),
                    is_current=work_data.get("is_current", False),
                    description=work_data.get("description", ""),
                    responsibilities=work_data.get("responsibilities", []),
                    achievements=work_data.get("achievements", []),
                )

                # Link skills used in this work experience
                skills_used = work_data.get("skills_used", [])
                for skill_name in skills_used:
                    if skill_name:
                        skill, _ = Skill.objects.get_or_create(
                            name__iexact=skill_name.strip(),
                            defaults={
                                "name": skill_name.strip(),
                                "category": default_category,
                                "skill_type": "TECHNICAL",
                            },
                        )
                        work_exp.skills_used.add(skill)

                work_records.append(work_exp)
            except Exception as e:
                print(f"Error creating work experience record: {str(e)}")
                continue

        return work_records

    def create_project_records(self, profile) -> list:
        """
        Create project records from parsed data

        Args:
            profile: UserProfile instance

        Returns:
            List of created Project instances
        """
        from apps.profiles.models import Project, Skill, SkillCategory

        projects = []

        # Get or create default category
        default_category, _ = SkillCategory.objects.get_or_create(
            name="General",
            defaults={"description": "General skills"},
        )

        for proj_data in self.parsed_data.get("projects", []):
            try:
                # Ensure URLs are never None
                project_url = proj_data.get("project_url") or ""
                github_url = proj_data.get("github_url") or ""
                demo_url = proj_data.get("demo_url") or ""

                if project_url is None:
                    project_url = ""
                if github_url is None:
                    github_url = ""
                if demo_url is None:
                    demo_url = ""

                project = Project.objects.create(
                    profile=profile,
                    title=proj_data.get("title", ""),
                    project_type=proj_data.get("project_type", "PERSONAL"),
                    description=proj_data.get("description", ""),
                    technologies_used=proj_data.get("technologies_used", []),
                    project_url=project_url,
                    github_url=github_url,
                    demo_url=demo_url,
                    start_date=self._parse_date(proj_data.get("start_date")) or "2000-01-01",
                    end_date=self._parse_date(proj_data.get("end_date")),
                    is_ongoing=proj_data.get("is_ongoing", False),
                )

                # Link skills demonstrated in this project
                skills_demonstrated = proj_data.get("skills_demonstrated", [])
                for skill_name in skills_demonstrated:
                    if skill_name:
                        skill, _ = Skill.objects.get_or_create(
                            name__iexact=skill_name.strip(),
                            defaults={
                                "name": skill_name.strip(),
                                "category": default_category,
                                "skill_type": "TECHNICAL",
                            },
                        )
                        project.skills_demonstrated.add(skill)

                projects.append(project)
            except Exception as e:
                print(f"Error creating project record: {str(e)}")
                continue

        return projects

    def create_certification_records(self, profile) -> list:
        """
        Create certification records from parsed data

        Args:
            profile: UserProfile instance

        Returns:
            List of created Certification instances
        """
        from apps.profiles.models import Certification, Skill, SkillCategory

        certifications = []

        # Get or create default category
        default_category, _ = SkillCategory.objects.get_or_create(
            name="General",
            defaults={"description": "General skills"},
        )

        for cert_data in self.parsed_data.get("certifications", []):
            try:
                certification = Certification.objects.create(
                    profile=profile,
                    name=cert_data.get("name", ""),
                    issuing_organization=cert_data.get("issuing_organization", ""),
                    credential_id=cert_data.get("credential_id", ""),
                    credential_url=cert_data.get("credential_url", ""),
                    issue_date=self._parse_date(cert_data.get("issue_date")) or "2000-01-01",
                    expiry_date=self._parse_date(cert_data.get("expiry_date")),
                    does_not_expire=cert_data.get("does_not_expire", False),
                )

                # Link skills validated by this certification
                skills_validated = cert_data.get("skills_validated", [])
                for skill_name in skills_validated:
                    if skill_name:
                        skill, _ = Skill.objects.get_or_create(
                            name__iexact=skill_name.strip(),
                            defaults={
                                "name": skill_name.strip(),
                                "category": default_category,
                                "skill_type": "TECHNICAL",
                            },
                        )
                        certification.skills_validated.add(skill)

                certifications.append(certification)
            except Exception as e:
                print(f"Error creating certification record: {str(e)}")
                continue

        return certifications

    def create_user_skills(self, profile) -> list:
        """
        Create user skills from parsed data

        Args:
            profile: UserProfile instance

        Returns:
            List of created UserSkill instances
        """
        from apps.profiles.models import Skill, UserSkill, SkillCategory

        user_skills = []

        # Get or create default category for unknown skills
        default_category, _ = SkillCategory.objects.get_or_create(
            name="General",
            defaults={"description": "General skills"},
        )

        for skill_data in self.parsed_data.get("skills", []):
            try:
                skill_name = skill_data.get("name", "").strip()
                if not skill_name:
                    continue

                # Get or create skill
                skill, _ = Skill.objects.get_or_create(
                    name__iexact=skill_name,
                    defaults={
                        "name": skill_name,
                        "category": default_category,
                        "skill_type": skill_data.get("category", "TECHNICAL"),
                    },
                )

                # Create or update user skill
                user_skill, _ = UserSkill.objects.update_or_create(
                    profile=profile,
                    skill=skill,
                    defaults={
                        "proficiency_level": skill_data.get("proficiency", "INTERMEDIATE"),
                        "years_of_experience": skill_data.get("years_of_experience", 0),
                    },
                )
                user_skills.append(user_skill)

            except Exception as e:
                print(f"Error creating user skill: {str(e)}")
                continue

        return user_skills

    def build_complete_profile(self, resume_file=None, resume_text: str = None) -> Dict[str, Any]:
        """
        Build complete user profile from parsed resume data

        Creates/updates:
        - UserProfile
        - Education records
        - Work experience records
        - Projects
        - Certifications
        - User skills

        Args:
            resume_file: Uploaded resume file (optional)
            resume_text: Extracted resume text

        Returns:
            Dictionary containing created objects
        """
        # Create/update profile
        profile = self.create_or_update_profile(resume_file, resume_text)

        # Create related records
        education_records = self.create_education_records(profile)
        work_records = self.create_work_experience_records(profile)
        projects = self.create_project_records(profile)
        certifications = self.create_certification_records(profile)
        user_skills = self.create_user_skills(profile)

        return {
            "profile": profile,
            "education_records": education_records,
            "work_records": work_records,
            "projects": projects,
            "certifications": certifications,
            "user_skills": user_skills,
            "total_records_created": (
                len(education_records)
                + len(work_records)
                + len(projects)
                + len(certifications)
                + len(user_skills)
            ),
        }


class ProfileChatService:
    """
    Service for AI-powered chat about user profile and career
    """

    def __init__(self):
        """Initialize chat service with GPT-4"""
        model_provider = os.getenv("MODEL_PROVIDER")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        gemini_model = "gemini-2.0-flash-exp"
        openai_model = "gpt-4o"

        if model_provider == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            self.llm = ChatGoogleGenerativeAI(model=gemini_model, api_key=gemini_api_key)
        else:
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=0.7,  # Slightly higher for more conversational responses
                api_key=api_key,
            )

    def _format_profile_context(self, user, profile) -> str:
        """Format user profile data into context for AI"""
        context_parts = []

        # Basic info
        context_parts.append(f"User: {user.first_name} {user.last_name}")
        if user.email:
            context_parts.append(f"Email: {user.email}")

        # Profile details
        if profile.bio:
            context_parts.append(f"\nBio: {profile.bio}")
        if profile.current_title:
            context_parts.append(f"Current Title: {profile.current_title}")
        if profile.current_company:
            context_parts.append(f"Current Company: {profile.current_company}")
        if profile.career_goal:
            context_parts.append(f"Career Goal: {profile.career_goal}")
        if profile.years_of_experience:
            context_parts.append(f"Years of Experience: {profile.years_of_experience}")

        # Education
        education = profile.education_records.all()
        if education:
            context_parts.append("\nEducation:")
            for edu in education:
                context_parts.append(
                    f"  - {edu.degree} in {edu.field_of_study or 'N/A'} from {edu.institution} ({edu.start_date.year}-{edu.end_date.year if edu.end_date else 'Present'})"
                )

        # Work Experience
        work_exp = profile.work_experiences.all().order_by('-start_date')
        if work_exp:
            context_parts.append("\nWork Experience:")
            for work in work_exp:
                end_year = work.end_date.year if work.end_date else "Present"
                context_parts.append(
                    f"  - {work.job_title} at {work.company} ({work.start_date.year}-{end_year})"
                )
                if work.description:
                    context_parts.append(f"    {work.description[:200]}")

        # Projects
        projects = profile.projects.all()
        if projects:
            context_parts.append("\nProjects:")
            for proj in projects:
                tech = ", ".join(proj.technologies_used) if proj.technologies_used else "N/A"
                context_parts.append(f"  - {proj.title} ({proj.project_type})")
                context_parts.append(f"    Technologies: {tech}")

        # Certifications
        certs = profile.certifications.all()
        if certs:
            context_parts.append("\nCertifications:")
            for cert in certs:
                context_parts.append(
                    f"  - {cert.name} from {cert.issuing_organization} (Issued: {cert.issue_date})"
                )

        # Skills
        try:
            from apps.skills.models import UserSkill
            user_skills = UserSkill.objects.filter(user=user).select_related('skill')
            if user_skills:
                context_parts.append("\nSkills:")
                for us in user_skills:
                    context_parts.append(f"  - {us.skill.name} ({us.proficiency_level})")
        except:
            pass

        return "\n".join(context_parts)

    def chat_about_profile(self, user, profile, message: str) -> str:
        """
        Generate AI response about user's profile

        Args:
            user: Django User object
            profile: UserProfile object
            message: User's question/message

        Returns:
            AI-generated response
        """
        # Format profile context
        profile_context = self._format_profile_context(user, profile)

        # Create chat prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful career advisor and resume expert for CareerCraft.
You help users understand their profile, improve their resume, and make career decisions.

Be conversational, supportive, and provide actionable advice. Use the user's profile data to give personalized recommendations.

If the user's profile is incomplete, gently suggest what they should add to strengthen their profile.

User Profile Data:
{profile_context}
"""),
            ("human", "{message}"),
        ])

        # Generate response
        chain = prompt | self.llm
        result = chain.invoke({
            "profile_context": profile_context,
            "message": message,
        })

        return result.content

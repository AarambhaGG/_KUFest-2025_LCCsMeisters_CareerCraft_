"""
LangChain-powered service for job eligibility analysis
"""

import json
import os
from typing import Dict, Any, List, Optional
from decimal import Decimal

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from django.conf import settings
from apps.users.models import User
from apps.profiles.models import (
    UserProfile,
    WorkExperience,
    Education,
    Certification,
    UserSkill,
    Skill,
    SkillCategory,
)
from .models import Job, JobEligibilityAnalysis


class DreamJobParser:
    """
    Service for parsing dream job descriptions using GPT-4
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

    def parse_job_description(self, job_description: str) -> Dict[str, Any]:
        """
        Parse a job description (pasted from job board or described by user)
        using GPT-4 to extract structured requirements

        Args:
            job_description: Raw job description text or user's dream job description

        Returns:
            Structured dictionary with job requirements
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert job description analyzer. Extract structured information from job postings.

Your task is to analyze the job description and extract the following information in JSON format:

{{
  "job_title": "Job title",
  "company_name": "Company name (or 'Not Specified' if user described dream job)",
  "job_type": "FULL_TIME/PART_TIME/CONTRACT/FREELANCE",
  "experience_level": "ENTRY/JUNIOR/MID/SENIOR/LEAD/EXECUTIVE",
  "location": "City, State/Country",
  "is_remote": <true/false>,
  "remote_policy": "FULLY_REMOTE/HYBRID/ON_SITE",
  "description": "Brief job description/summary",
  "responsibilities": ["Responsibility 1", "Responsibility 2", ...],
  "required_skills": [
    {{
      "name": "Skill name",
      "requirement_type": "MUST_HAVE",
      "minimum_proficiency": "BEGINNER/INTERMEDIATE/ADVANCED/EXPERT",
      "years_required": <number or null>
    }}
  ],
  "preferred_skills": [
    {{
      "name": "Skill name",
      "requirement_type": "NICE_TO_HAVE",
      "minimum_proficiency": "BEGINNER/INTERMEDIATE/ADVANCED/EXPERT"
    }}
  ],
  "education_requirements": {{
    "degree_level": "HIGH_SCHOOL/ASSOCIATE/BACHELOR/MASTER/PHD",
    "field_of_study": "Preferred field (or null)",
    "is_required": <true/false>
  }},
  "min_years_experience": <number or null>,
  "max_years_experience": <number or null>,
  "min_salary": <number or null>,
  "max_salary": <number or null>,
  "salary_currency": "USD/EUR/GBP etc.",
  "benefits": ["Benefit 1", "Benefit 2", ...],
  "company_culture": "Description of culture/values",
  "industry": "Industry sector"
}}

IMPORTANT RULES:
1. Extract ALL available information, even if incomplete
2. For missing fields, use null or empty arrays
3. Categorize skills into MUST_HAVE (required) vs NICE_TO_HAVE (preferred)
4. Infer minimum proficiency from context (e.g., "5+ years Python" = ADVANCED)
5. Extract years of experience requirements for specific skills when mentioned
6. Standardize skill names (e.g., "JavaScript" not "javascript")
7. If user describes a dream job without specifics, infer reasonable requirements
8. For salary, extract ranges when provided
9. Return ONLY valid JSON, no additional text

If information is ambiguous or missing, make reasonable inferences based on context and industry standards.""",
                ),
                ("human", "Job description:\n\n{job_description}"),
            ]
        )

        try:
            # Invoke GPT-4 to parse job description
            chain = prompt | self.llm
            response = chain.invoke({"job_description": job_description})

            # Extract JSON from response
            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            # Parse JSON
            parsed_data = json.loads(content.strip())

            return parsed_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GPT-4 response as JSON: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing job description with GPT-4: {str(e)}")

    def create_temporary_job(self, parsed_data: Dict[str, Any]) -> Job:
        """
        Create a temporary (unsaved) Job object from parsed data for analysis

        Args:
            parsed_data: Parsed job description data from parse_job_description

        Returns:
            Unsaved Job instance
        """
        # Map remote policy to correct values (FULLY_REMOTE -> REMOTE, HYBRID, ON_SITE -> ONSITE)
        remote_policy = parsed_data.get("remote_policy", "REMOTE")
        if remote_policy == "FULLY_REMOTE":
            remote_policy = "REMOTE"
        elif remote_policy == "ON_SITE":
            remote_policy = "ONSITE"

        # Build job kwargs, only include salary_currency if it has a value
        job_kwargs = {
            "title": parsed_data.get("job_title", "Unknown Title"),
            "company_name": parsed_data.get("company_name", "Dream Company"),
            "company_description": parsed_data.get("company_culture", ""),
            "job_type": parsed_data.get("job_type", "FULL_TIME"),
            "experience_level": parsed_data.get("experience_level", "MID"),
            "location": parsed_data.get("location", "Remote"),
            "is_remote": parsed_data.get("is_remote", True),
            "remote_policy": remote_policy,
            "description": parsed_data.get("description", ""),
            "responsibilities": (
                "\n".join(
                    [f"• {resp}" for resp in parsed_data.get("responsibilities", [])]
                )
                if isinstance(parsed_data.get("responsibilities"), list)
                else str(parsed_data.get("responsibilities", ""))
            ),
            "requirements": "\n".join(
                [
                    f"• {req.get('name', req) if isinstance(req, dict) else req}"
                    for req in parsed_data.get("required_skills", [])
                ]
            ),
            "salary_min": parsed_data.get("min_salary"),
            "salary_max": parsed_data.get("max_salary"),
            "source_url": "",  # Empty for temporary jobs
            "parsed_skills": parsed_data.get("required_skills", [])
            + parsed_data.get("preferred_skills", []),
            "parsed_requirements": parsed_data,
        }

        # Only add salary_currency if provided (let model default handle it otherwise)
        if parsed_data.get("salary_currency"):
            job_kwargs["salary_currency"] = parsed_data.get("salary_currency")

        job = Job(**job_kwargs)

        # Note: Job is NOT saved to database
        # This is intentional - it's a temporary object for analysis only
        return job


class JobEligibilityAnalyzer:
    """
    LangChain-based service for analyzing user eligibility for job postings
    """

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the analyzer with specified LLM model

        Args:
            model_name: OpenAI model to use (default: gpt-4)
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,  # Lower temperature for more consistent analysis
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def _gather_user_context(self, user: User) -> Dict[str, Any]:
        """
        Gather comprehensive user context for analysis

        Args:
            user: User instance

        Returns:
            Dictionary containing user profile, skills, experience, etc.
        """
        context = {
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
        }

        # Profile data
        try:
            profile = user.profile
            context["profile"] = {
                "bio": profile.bio,
                "current_title": profile.current_title,
                "current_company": profile.current_company,
                "years_of_experience": float(profile.years_of_experience),
                "career_goal": profile.career_goal,
                "target_roles": profile.target_roles,
                "industry": profile.industry,
                "domain_expertise": profile.domain_expertise,
            }
        except UserProfile.DoesNotExist:
            context["profile"] = {}

        # Job preferences
        try:
            prefs = user.preferences
            context["job_preferences"] = {
                "preferred_job_types": prefs.preferred_job_types,
                "preferred_locations": prefs.preferred_locations,
                "remote_preference": prefs.remote_preference,
                "salary_expectation_min": prefs.salary_expectation_min,
                "salary_expectation_max": prefs.salary_expectation_max,
                "willing_to_relocate": prefs.willing_to_relocate,
            }
        except:
            context["job_preferences"] = {}

        # Skills
        try:
            user_skills = UserSkill.objects.filter(profile=user.profile).select_related(
                "skill"
            )
        except UserProfile.DoesNotExist:
            user_skills = []
        context["skills"] = [
            {
                "name": us.skill.name,
                "proficiency_level": us.proficiency_level,
                "years_of_experience": (
                    float(us.years_of_experience) if us.years_of_experience else 0
                ),
                "is_verified": us.is_verified,
                "verified_by": us.verified_by,
                "last_used": us.last_used.isoformat() if us.last_used else None,
            }
            for us in user_skills
        ]

        # Work Experience
        try:
            work_experiences = WorkExperience.objects.filter(
                profile=user.profile
            ).prefetch_related("skills_used")
            context["work_experience"] = [
                {
                    "job_title": exp.job_title,
                    "company": exp.company,
                    "employment_type": exp.employment_type,
                    "location": exp.location,
                    "is_remote": exp.is_remote,
                    "start_date": exp.start_date.isoformat(),
                    "end_date": exp.end_date.isoformat() if exp.end_date else "Present",
                    "is_current": exp.is_current,
                    "description": exp.description,
                    "responsibilities": exp.responsibilities,
                    "achievements": exp.achievements,
                    "skills_used": [skill.name for skill in exp.skills_used.all()],
                }
                for exp in work_experiences
            ]
        except:
            context["work_experience"] = []

        # Education
        try:
            education = Education.objects.filter(profile=user.profile)
            context["education"] = [
                {
                    "institution": edu.institution,
                    "degree": edu.degree,
                    "degree_level": edu.degree_level,
                    "field_of_study": edu.field_of_study,
                    "start_date": edu.start_date.isoformat(),
                    "end_date": edu.end_date.isoformat() if edu.end_date else "Present",
                    "is_current": edu.is_current,
                }
                for edu in education
            ]
        except:
            context["education"] = []

        # Certifications
        try:
            certifications = Certification.objects.filter(
                profile=user.profile
            ).prefetch_related("skills_validated")
            context["certifications"] = [
                {
                    "name": cert.name,
                    "issuing_organization": cert.issuing_organization,
                    "issue_date": cert.issue_date.isoformat(),
                    "skills_validated": [
                        skill.name for skill in cert.skills_validated.all()
                    ],
                }
                for cert in certifications
            ]
        except:
            context["certifications"] = []

        return context

    def _gather_job_context(self, job: Job) -> Dict[str, Any]:
        """
        Gather comprehensive job context for analysis

        Args:
            job: Job instance

        Returns:
            Dictionary containing job details, requirements, etc.
        """
        context = {
            "title": job.title,
            "company_name": job.company_name,
            "company_description": job.company_description,
            "description": job.description,
            "responsibilities": job.responsibilities,
            "requirements": job.requirements,
            "nice_to_have": job.nice_to_have,
            "job_type": job.job_type,
            "experience_level": job.experience_level,
            "location": job.location,
            "is_remote": job.is_remote,
            "remote_policy": job.remote_policy,
            "salary_min": job.salary_min,
            "salary_max": job.salary_max,
            "salary_currency": job.salary_currency,
            "salary_period": job.salary_period,
        }

        # Required skills
        skill_requirements = job.skill_requirements.all().select_related("skill")
        context["required_skills"] = [
            {
                "skill_name": req.skill.name,
                "requirement_type": req.requirement_type,
                "minimum_proficiency": req.minimum_proficiency,
                "years_required": req.years_required,
                "weight": req.weight,
            }
            for req in skill_requirements
        ]

        # Parsed data
        context["parsed_skills"] = job.parsed_skills
        context["parsed_requirements"] = job.parsed_requirements

        return context

    def _create_prompt(
        self,
        user_context: Dict[str, Any],
        job_context: Dict[str, Any],
        additional_context: str = "",
    ) -> str:
        """
        Create the prompt for job eligibility analysis

        Args:
            user_context: User profile and experience data
            job_context: Job posting details
            additional_context: Additional context from user

        Returns:
            Prompt string for analysis
        """
        additional_section = ""
        if additional_context:
            additional_section = (
                f"\n\n**ADDITIONAL CONTEXT FROM CANDIDATE:**\n{additional_context}"
            )

        prompt = f"""You are an expert career counselor and recruiter analyzing whether a candidate is eligible for a job position.

**CANDIDATE PROFILE:**
{json.dumps(user_context, indent=2)}

**JOB POSTING:**
{json.dumps(job_context, indent=2)}
{additional_section}

**YOUR TASK:**
Analyze the candidate's eligibility for this job position. Consider:
1. Skills match (technical and soft skills)
2. Experience level and years of experience
3. Education and certifications
4. Work history and achievements
5. Career trajectory and goals
6. Location and remote work preferences
7. Salary expectations vs. job offer
8. Culture fit and soft skills
9. Domain knowledge and industry expertise
10. Overall readiness and preparation needed

Provide a comprehensive, multi-dimensional analysis with detailed metrics across all dimensions.

Be honest but constructive. If there are significant gaps, explain them clearly but also provide actionable guidance on how to bridge them.

**IMPORTANT: Return your response as a valid JSON object with the following structure:**
{{
    "eligibility_level": "EXCELLENT/GOOD/FAIR/POOR",
    "match_score": 0-100,
    "analysis_summary": "2-3 sentence summary",
    "strengths": ["strength 1", "strength 2", ...],
    "gaps": ["gap 1", "gap 2", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...],
    "matching_skills": ["skill 1", "skill 2", ...],
    "missing_skills": ["skill 1", "skill 2", ...],
    "skill_gaps": [
        {{
            "skill_name": "Skill name",
            "required_level": "BEGINNER/INTERMEDIATE/ADVANCED/EXPERT",
            "current_level": "BEGINNER/INTERMEDIATE/ADVANCED/EXPERT",
            "gap_severity": "LOW/MEDIUM/HIGH/CRITICAL",
            "priority": "LOW/MEDIUM/HIGH/CRITICAL",
            "estimated_time_to_learn": "e.g., 2-3 months"
        }}
    ],
    "skills_match_score": 0-100,
    "experience_match_score": 0-100,
    "education_match_score": 0-100,
    "culture_fit_score": 0-100,
    "location_match_score": 0-100,
    "salary_match_score": 0-100,
    "technical_skills_score": 0-100,
    "soft_skills_score": 0-100,
    "domain_knowledge_score": 0-100,
    "experience_match": "Explanation of how experience matches",
    "experience_gap_years": 0.0 or null,
    "years_of_experience_required": 0.0 or null,
    "years_of_experience_user": 0.0 or null,
    "readiness_percentage": 0-100,
    "estimated_preparation_time": "e.g., 3-6 months, or 'Ready now'",
    "confidence_level": "VERY_HIGH/HIGH/MEDIUM/LOW/VERY_LOW",
    "next_steps": [
        "Concrete action step 1",
        "Concrete action step 2",
        ...
    ],
    "priority_improvements": [
        {{
            "area": "Skill/Experience area",
            "current_state": "Current level/state",
            "target_state": "Desired level/state",
            "impact": "HIGH/MEDIUM/LOW",
            "effort": "HIGH/MEDIUM/LOW",
            "timeline": "Estimated time needed"
        }}
    ],
    "learning_resources": [
        {{
            "resource_type": "COURSE/CERTIFICATION/BOOK/PROJECT/PRACTICE",
            "title": "Resource title",
            "description": "What this resource covers",
            "estimated_duration": "Time to complete",
            "priority": "HIGH/MEDIUM/LOW"
        }}
    ]
}}

**SCORING GUIDELINES:**
- skills_match_score: How well user's skills match job requirements (0-100)
- experience_match_score: How well years and type of experience match (0-100)
- education_match_score: How well education matches requirements (0-100)
- culture_fit_score: Estimated culture/values alignment based on profile (0-100)
- location_match_score: Location/remote preference match (0-100)
- salary_match_score: Salary expectation vs offering alignment (0-100)
- technical_skills_score: Technical competencies only (0-100)
- soft_skills_score: Communication, leadership, teamwork, etc. (0-100)
- domain_knowledge_score: Industry/domain expertise (0-100)
- readiness_percentage: Overall readiness to apply and succeed (0-100)

Return ONLY the JSON object, no additional text.
"""
        return prompt

    def analyze_eligibility(
        self, user: User, job: Job, additional_context: str = ""
    ) -> JobEligibilityAnalysis:
        """
        Perform comprehensive job eligibility analysis

        Args:
            user: User to analyze
            job: Job to analyze for
            additional_context: Additional context provided by user

        Returns:
            JobEligibilityAnalysis instance with analysis results
        """
        # Gather context
        user_context = self._gather_user_context(user)
        job_context = self._gather_job_context(job)

        # Create prompt
        prompt = self._create_prompt(user_context, job_context, additional_context)

        # Run analysis
        response = self.llm.invoke(prompt)

        # Extract content from response
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse JSON response
        try:
            # Try to find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            # Fallback: create a basic response
            result = {
                "eligibility_level": "FAIR",
                "match_score": 50,
                "analysis_summary": f"Analysis could not be parsed properly. Raw response available in full_analysis field.",
                "strengths": [],
                "gaps": [],
                "recommendations": [],
                "matching_skills": [],
                "missing_skills": [],
                "experience_match": "Unable to parse experience match",
                "experience_gap_years": None,
            }

        # Helper function to safely convert to Decimal
        def to_decimal(value):
            if value and value != "null":
                try:
                    return Decimal(str(value))
                except:
                    return None
            return None

        # Helper function to safely convert to int with default
        def to_int(value, default=0):
            if value is not None and value != "null":
                try:
                    return int(value)
                except:
                    return default
            return default

        # Ensure lists are properly formatted
        def ensure_list(value):
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, list) else [value]
                except:
                    return [value] if value else []
            return []

        # Parse all numeric fields
        exp_gap = to_decimal(result.get("experience_gap_years"))
        years_required = to_decimal(result.get("years_of_experience_required"))
        years_user = to_decimal(result.get("years_of_experience_user"))

        # Parse all score fields with defaults
        skills_match_score = to_int(result.get("skills_match_score"), 0)
        experience_match_score = to_int(result.get("experience_match_score"), 0)
        education_match_score = to_int(result.get("education_match_score"), 0)
        culture_fit_score = to_int(result.get("culture_fit_score"), 0)
        location_match_score = to_int(result.get("location_match_score"), 0)
        salary_match_score = to_int(result.get("salary_match_score"), 0)
        technical_skills_score = to_int(result.get("technical_skills_score"), 0)
        soft_skills_score = to_int(result.get("soft_skills_score"), 0)
        domain_knowledge_score = to_int(result.get("domain_knowledge_score"), 0)
        readiness_percentage = to_int(result.get("readiness_percentage"), 0)

        # Parse confidence level with validation
        confidence_level = result.get("confidence_level", "MEDIUM")
        valid_confidence = ["VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW"]
        if confidence_level not in valid_confidence:
            confidence_level = "MEDIUM"

        # Create analysis record with all new fields
        analysis = JobEligibilityAnalysis.objects.create(
            user=user,
            job=job,
            additional_context=additional_context,
            # Core analysis
            eligibility_level=result.get("eligibility_level", "FAIR"),
            match_score=int(result.get("match_score", 50)),
            analysis_summary=result.get("analysis_summary", ""),
            strengths=ensure_list(result.get("strengths", [])),
            gaps=ensure_list(result.get("gaps", [])),
            recommendations=ensure_list(result.get("recommendations", [])),
            # Skills analysis
            matching_skills=ensure_list(result.get("matching_skills", [])),
            missing_skills=ensure_list(result.get("missing_skills", [])),
            skill_gaps=ensure_list(result.get("skill_gaps", [])),
            # Detailed match metrics
            skills_match_score=skills_match_score,
            experience_match_score=experience_match_score,
            education_match_score=education_match_score,
            culture_fit_score=culture_fit_score,
            location_match_score=location_match_score,
            salary_match_score=salary_match_score,
            # Categorized scores
            technical_skills_score=technical_skills_score,
            soft_skills_score=soft_skills_score,
            domain_knowledge_score=domain_knowledge_score,
            # Experience details
            experience_match=result.get("experience_match", ""),
            experience_gap_years=exp_gap,
            years_of_experience_required=years_required,
            years_of_experience_user=years_user,
            # Readiness metrics
            readiness_percentage=readiness_percentage,
            estimated_preparation_time=result.get("estimated_preparation_time", ""),
            confidence_level=confidence_level,
            # Next steps & actionability
            next_steps=ensure_list(result.get("next_steps", [])),
            priority_improvements=ensure_list(result.get("priority_improvements", [])),
            learning_resources=ensure_list(result.get("learning_resources", [])),
            # Metadata
            full_analysis=response_text,
            llm_model=self.model_name,
            token_usage=0,  # Can be enhanced to track actual token usage
        )

        return analysis

    def reanalyze_with_context(
        self, analysis: JobEligibilityAnalysis, additional_context: str
    ) -> JobEligibilityAnalysis:
        """
        Re-run analysis with additional context from user

        Args:
            analysis: Previous analysis
            additional_context: New context to add

        Returns:
            New JobEligibilityAnalysis instance
        """
        combined_context = analysis.additional_context
        if combined_context:
            combined_context += "\n\n" + additional_context
        else:
            combined_context = additional_context

        return self.analyze_eligibility(
            user=analysis.user, job=analysis.job, additional_context=combined_context
        )

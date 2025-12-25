"""
Views for Profiles app
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema, extend_schema_view

from .models import (
    UserProfile,
    Education,
    WorkExperience,
    Project,
    Certification,
)
from .serializers import (
    UserProfileSerializer,
    EducationSerializer,
    WorkExperienceSerializer,
    ProjectSerializer,
    CertificationSerializer,
    CompleteProfileSerializer,
)
from .services import ResumeParserService, ProfileBuilderService, ProfileChatService


@extend_schema_view(
    retrieve=extend_schema(
        tags=['Profile'],
        summary='Get my profile',
        description='Get the current user\'s profile',
    ),
    update=extend_schema(
        tags=['Profile'],
        summary='Update my profile',
        description='Update the current user\'s profile',
    ),
    partial_update=extend_schema(
        tags=['Profile'],
        summary='Partially update my profile',
        description='Partially update the current user\'s profile',
    ),
)
class UserProfileViewSet(viewsets.ModelViewSet):
    """
    ViewSet for user profile management
    """
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    http_method_names = ['get', 'post', 'put', 'patch']

    def get_queryset(self):
        """Get or create profile for current user"""
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        return UserProfile.objects.filter(user=self.request.user)

    def get_object(self):
        """Get the current user's profile"""
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        return profile

    @extend_schema(
        tags=['Profile'],
        summary='Get complete profile',
        description='Get complete profile with all related data (education, experience, projects, certifications)',
    )
    @action(detail=False, methods=['get'])
    def complete(self, request):
        """
        Get complete profile with all related data
        """
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        serializer = CompleteProfileSerializer(profile)
        return Response(serializer.data)

    @extend_schema(
        tags=['Profile'],
        summary='Get profile statistics',
        description='Get statistics about the user\'s profile completion',
    )
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get profile statistics
        """
        profile, created = UserProfile.objects.get_or_create(user=request.user)

        # Calculate profile completion
        completion_fields = [
            profile.bio,
            profile.current_title,
            profile.current_company,
            profile.career_goal,
            profile.linkedin_url,
            profile.github_url,
        ]
        completed = sum(1 for field in completion_fields if field)
        completion_percentage = int((completed / len(completion_fields)) * 100)

        stats = {
            'profile_completion': completion_percentage,
            'has_resume': bool(profile.resume),
            'education_count': profile.education_records.count(),
            'work_experience_count': profile.work_experiences.count(),
            'projects_count': profile.projects.count(),
            'certifications_count': profile.certifications.count(),
            'years_of_experience': float(profile.years_of_experience),
            'has_career_goal': bool(profile.career_goal),
            'target_roles_count': len(profile.target_roles) if profile.target_roles else 0,
        }

        return Response(stats)

    @extend_schema(
        tags=['Profile'],
        summary='Upload and parse resume',
        description='Upload a resume file (PDF or DOCX) and get AI-parsed structured data',
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'resume': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Resume file (PDF or DOCX)',
                    }
                },
                'required': ['resume'],
            }
        },
    )
    @action(detail=False, methods=['post'])
    def upload_resume(self, request):
        """
        Upload and parse resume using GPT-4

        Returns:
            - resume_text: Extracted text from resume
            - parsed_data: AI-parsed structured data
        """
        resume_file = request.FILES.get('resume')

        if not resume_file:
            return Response(
                {'error': 'No resume file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Parse resume with AI
            parser = ResumeParserService()
            result = parser.parse_resume_file(resume_file)

            return Response({
                'message': 'Resume parsed successfully',
                'resume_text': result['resume_text'],
                'parsed_data': result['parsed_data'],
            }, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Failed to parse resume: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        tags=['Profile'],
        summary='Build profile from parsed resume',
        description='Create complete user profile from AI-parsed resume data',
    )
    @action(detail=False, methods=['post'])
    def build_from_resume(self, request):
        """
        Build complete profile from parsed resume data

        Expected request data:
            - parsed_data: AI-parsed resume data from upload_resume endpoint
            - resume_file: (optional) Resume file to save

        Creates/updates:
            - UserProfile
            - Education records
            - Work experience records
            - Projects
            - Certifications
            - User skills
        """
        parsed_data = request.data.get('parsed_data')

        if not parsed_data:
            return Response(
                {'error': 'parsed_data is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Build profile from parsed data
            builder = ProfileBuilderService(request.user, parsed_data)
            result = builder.build_complete_profile(
                resume_file=request.FILES.get('resume'),
                resume_text=request.data.get('resume_text', '')
            )

            # Return profile stats
            profile = result['profile']
            return Response({
                'message': 'Profile built successfully',
                'profile_id': profile.id,
                'records_created': {
                    'education': len(result['education_records']),
                    'work_experience': len(result['work_records']),
                    'projects': len(result['projects']),
                    'certifications': len(result['certifications']),
                    'skills': len(result['user_skills']),
                    'total': result['total_records_created'],
                },
                'profile_completion': self._calculate_completion(profile),
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {'error': f'Failed to build profile: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        tags=['Profile'],
        summary='Complete onboarding: Upload & Build',
        description='All-in-one endpoint: Upload resume, parse with AI, and build complete profile',
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'resume': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Resume file (PDF or DOCX)',
                    }
                },
                'required': ['resume'],
            }
        },
    )
    @action(detail=False, methods=['post'])
    def onboard(self, request):
        """
        Complete onboarding flow in one request:
        1. Upload resume
        2. Parse with GPT-4
        3. Build complete profile

        Returns complete profile with all created records
        """
        resume_file = request.FILES.get('resume')

        if not resume_file:
            return Response(
                {'error': 'No resume file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Step 1 & 2: Parse resume
            parser = ResumeParserService()
            parse_result = parser.parse_resume_file(resume_file)

            # Step 3: Build profile
            builder = ProfileBuilderService(request.user, parse_result['parsed_data'])

            # Save the resume file
            # Reset file pointer after parsing
            resume_file.seek(0)

            build_result = builder.build_complete_profile(
                resume_file=resume_file,
                resume_text=parse_result['resume_text']
            )

            # Get complete profile data
            profile = build_result['profile']
            serializer = CompleteProfileSerializer(profile)

            return Response({
                'message': 'Onboarding completed successfully',
                'onboarding_summary': {
                    'time_to_complete': '~2 minutes',
                    'records_created': {
                        'education': len(build_result['education_records']),
                        'work_experience': len(build_result['work_records']),
                        'projects': len(build_result['projects']),
                        'certifications': len(build_result['certifications']),
                        'skills': len(build_result['user_skills']),
                        'total': build_result['total_records_created'],
                    },
                    'profile_completion': self._calculate_completion(profile),
                },
                'profile': serializer.data,
            }, status=status.HTTP_201_CREATED)

        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Onboarding failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        tags=['Profile'],
        summary='Chat about profile and resume',
        description='Ask questions about your profile, get career advice, and receive personalized recommendations',
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'description': 'User message/question about their profile',
                    },
                },
                'required': ['message'],
            }
        },
    )
    @action(detail=False, methods=['post'])
    def chat(self, request):
        """
        Chat with AI about your profile and resume

        Ask questions like:
        - "What are my strongest skills?"
        - "How can I improve my resume?"
        - "What jobs should I apply for?"
        - "What skills should I learn next?"
        """
        message = request.data.get('message')

        if not message:
            return Response(
                {'error': 'message is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get or create user profile
            profile, created = UserProfile.objects.get_or_create(user=request.user)

            # Initialize chat service
            chat_service = ProfileChatService()

            # Get response from AI
            response = chat_service.chat_about_profile(
                user=request.user,
                profile=profile,
                message=message
            )

            return Response({
                'message': message,
                'response': response,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': f'Chat failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _calculate_completion(self, profile):
        """Helper to calculate profile completion percentage"""
        completion_fields = [
            profile.bio,
            profile.current_title,
            profile.current_company,
            profile.career_goal,
            profile.linkedin_url,
            profile.github_url,
        ]
        completed = sum(1 for field in completion_fields if field)

        # Add bonus for having related records
        if profile.education_records.exists():
            completed += 1
        if profile.work_experiences.exists():
            completed += 1
        if profile.projects.exists():
            completed += 0.5
        if profile.certifications.exists():
            completed += 0.5

        total_fields = len(completion_fields) + 3
        return int((completed / total_fields) * 100)


@extend_schema_view(
    list=extend_schema(
        tags=['Education'],
        summary='List my education records',
        description='Get all education records for the current user',
    ),
    create=extend_schema(
        tags=['Education'],
        summary='Add education record',
        description='Add a new education record',
    ),
    retrieve=extend_schema(
        tags=['Education'],
        summary='Get education details',
        description='Get details of a specific education record',
    ),
    update=extend_schema(
        tags=['Education'],
        summary='Update education record',
        description='Update an education record',
    ),
    partial_update=extend_schema(
        tags=['Education'],
        summary='Partially update education',
        description='Partially update an education record',
    ),
    destroy=extend_schema(
        tags=['Education'],
        summary='Delete education record',
        description='Delete an education record',
    ),
)
class EducationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for education records
    """
    serializer_class = EducationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get education records for current user's profile"""
        try:
            profile = self.request.user.profile
            return Education.objects.filter(profile=profile)
        except UserProfile.DoesNotExist:
            return Education.objects.none()

    def perform_create(self, serializer):
        """Create education record for current user's profile"""
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        serializer.save(profile=profile)


@extend_schema_view(
    list=extend_schema(
        tags=['Work Experience'],
        summary='List my work experiences',
        description='Get all work experience records for the current user',
    ),
    create=extend_schema(
        tags=['Work Experience'],
        summary='Add work experience',
        description='Add a new work experience record',
    ),
    retrieve=extend_schema(
        tags=['Work Experience'],
        summary='Get work experience details',
        description='Get details of a specific work experience',
    ),
    update=extend_schema(
        tags=['Work Experience'],
        summary='Update work experience',
        description='Update a work experience record',
    ),
    partial_update=extend_schema(
        tags=['Work Experience'],
        summary='Partially update work experience',
        description='Partially update a work experience record',
    ),
    destroy=extend_schema(
        tags=['Work Experience'],
        summary='Delete work experience',
        description='Delete a work experience record',
    ),
)
class WorkExperienceViewSet(viewsets.ModelViewSet):
    """
    ViewSet for work experience records
    """
    serializer_class = WorkExperienceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get work experiences for current user's profile"""
        try:
            profile = self.request.user.profile
            return WorkExperience.objects.filter(profile=profile).prefetch_related('skills_used')
        except UserProfile.DoesNotExist:
            return WorkExperience.objects.none()

    def perform_create(self, serializer):
        """Create work experience for current user's profile"""
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        serializer.save(profile=profile)


@extend_schema_view(
    list=extend_schema(
        tags=['Projects'],
        summary='List my projects',
        description='Get all project records for the current user',
    ),
    create=extend_schema(
        tags=['Projects'],
        summary='Add project',
        description='Add a new project',
    ),
    retrieve=extend_schema(
        tags=['Projects'],
        summary='Get project details',
        description='Get details of a specific project',
    ),
    update=extend_schema(
        tags=['Projects'],
        summary='Update project',
        description='Update a project',
    ),
    partial_update=extend_schema(
        tags=['Projects'],
        summary='Partially update project',
        description='Partially update a project',
    ),
    destroy=extend_schema(
        tags=['Projects'],
        summary='Delete project',
        description='Delete a project',
    ),
)
class ProjectViewSet(viewsets.ModelViewSet):
    """
    ViewSet for project records
    """
    serializer_class = ProjectSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get projects for current user's profile"""
        try:
            profile = self.request.user.profile
            return Project.objects.filter(profile=profile).prefetch_related('skills_demonstrated')
        except UserProfile.DoesNotExist:
            return Project.objects.none()

    def perform_create(self, serializer):
        """Create project for current user's profile"""
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        serializer.save(profile=profile)


@extend_schema_view(
    list=extend_schema(
        tags=['Certifications'],
        summary='List my certifications',
        description='Get all certification records for the current user',
    ),
    create=extend_schema(
        tags=['Certifications'],
        summary='Add certification',
        description='Add a new certification',
    ),
    retrieve=extend_schema(
        tags=['Certifications'],
        summary='Get certification details',
        description='Get details of a specific certification',
    ),
    update=extend_schema(
        tags=['Certifications'],
        summary='Update certification',
        description='Update a certification',
    ),
    partial_update=extend_schema(
        tags=['Certifications'],
        summary='Partially update certification',
        description='Partially update a certification',
    ),
    destroy=extend_schema(
        tags=['Certifications'],
        summary='Delete certification',
        description='Delete a certification',
    ),
)
class CertificationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for certification records
    """
    serializer_class = CertificationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get certifications for current user's profile"""
        try:
            profile = self.request.user.profile
            return Certification.objects.filter(profile=profile).prefetch_related('skills_validated')
        except UserProfile.DoesNotExist:
            return Certification.objects.none()

    def perform_create(self, serializer):
        """Create certification for current user's profile"""
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        serializer.save(profile=profile)

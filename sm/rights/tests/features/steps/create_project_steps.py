from behave import given, when, then, step

class User(object):
    def __init__(self, name):
        self.name = name

class Project(object):
    def __init__(self, name, manager):
        self.name = name
        self.manager = manager

class ProjectDatabase(object):
    def __init__(self):
        self._projects = {}
    
    def add(self, project):
        if project.name in self._projects:
            raise Exception('project name already taken')
        self._projects[project.name] = project

    def __getitem__(self, name):
        return self._projects.get(name, None)

@given("existing projects")
def step_impl(ctx):
    ctx.projects = ProjectDatabase()
    for row in ctx.table:
        ctx.projects.add(Project(row['name'], User('Alice')))

@given("the name {name} has already been taken")
def step_impl(ctx, name):
    assert ctx.projects[name] is not None

@given("the name {name} has not yet been taken")
def step_impl(ctx, name):
    assert ctx.projects[name] is None

@when("I try to create a project named {name}")
def step_impl(ctx, name):
    ctx.user = User('Bob')
    ctx.project_name = name
    try:
        ctx.projects.add(Project(name, ctx.user))
        ctx.project_created = True
    except Exception as e:
        ctx.project_created = False
        ctx.project_creation_error = e

@then("I get an error message")
def step_impl(ctx):
    assert not ctx.project_created and ctx.project_creation_error

@then("the project is created")
def step_impl(ctx):
    assert ctx.projects[ctx.project_name] is not None

@then("I become that project's administrator")
def step_impl(ctx):
    proj = ctx.projects[ctx.project_name]
    assert proj.manager == ctx.user
